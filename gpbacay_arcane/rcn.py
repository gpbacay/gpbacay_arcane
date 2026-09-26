"""The ``.rcn`` model container for ARC 1.

One self-contained file holds everything needed to run ARC 1: architecture,
readout calibration, tokenizer, a baked device profile, and the weights.

Layout (little-endian)::

    0     header    128 bytes, fixed (see ``_HEADER``)
    128   metadata  compact UTF-8 JSON: config, tokenizer, profile, tensor directory
    ...   data      tensors, each 64-byte aligned, stored in their final layout

The header alone describes the model, so a runtime can inspect a file without
touching the weights. Tensor data is memory-mapped and read in place: ``f32``
and ``f16`` tensors are used straight from the mapping, and quantized tensors
are expanded in one vectorised step with no parsing.

Weight formats (``quant``):

* ``f16`` — half precision.
* ``rq8`` — 8-bit symmetric integers, one f16 scale per block of 32 values
  (8.5 bits per weight).
* ``rq4`` — 4-bit symmetric integers packed two per byte, one f16 scale per
  block of 32 values (4.5 bits per weight).

Tensors with fewer than ``MIN_QUANT_SIZE`` values (norms, biases, gates) always
stay f16, since quantizing them saves nothing and costs accuracy.
"""

from __future__ import annotations

import json
import mmap
import struct
import zlib
from typing import Any, Dict, Optional, Tuple

import numpy as np

MAGIC = b"RCN\x00"
VERSION = 1
HEADER_SIZE = 128
ALIGN = 64
BLOCK = 32
MIN_QUANT_SIZE = 4096
QUANTS = ("f16", "rq8", "rq4")

# magic, version, flags, quant, cycles, d_model, layers, heads, vocab, seq_len,
# tensor count, metadata offset/length, data offset/length, data crc32
_HEADER = struct.Struct("<4sHHBBHHHIHIQQQQI")
_QUANT_CODE = {q: i for i, q in enumerate(QUANTS)}
FLAG_TOKENIZER = 1


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def _quantize(x: np.ndarray, bits: int) -> Tuple[bytes, bytes]:
    """Block-wise symmetric quantization along the flattened tensor."""
    flat = x.astype(np.float32).ravel()
    pad = (-flat.size) % BLOCK
    blocks = np.pad(flat, (0, pad)).reshape(-1, BLOCK)
    qmax = (1 << (bits - 1)) - 1
    scale = np.abs(blocks).max(axis=1) / qmax
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(blocks / scale[:, None]), -qmax, qmax).astype(np.int8)
    if bits == 4:
        u = (q.ravel() & 0x0F).astype(np.uint8)
        q = (u[0::2] | (u[1::2] << 4)).astype(np.uint8)
    return q.tobytes(), scale.astype(np.float16).tobytes()


def _dequantize(buf: memoryview, entry: Dict[str, Any]) -> np.ndarray:
    size = int(np.prod(entry["shape"])) if entry["shape"] else 1
    n_blocks = -(-size // BLOCK)
    scale = np.frombuffer(buf, np.float16, n_blocks, entry["scale_offset"]).astype(np.float32)
    if entry["dtype"] == "rq8":
        q = np.frombuffer(buf, np.int8, n_blocks * BLOCK, entry["offset"]).astype(np.float32)
    else:
        packed = np.frombuffer(buf, np.uint8, n_blocks * BLOCK // 2, entry["offset"])
        lo = (packed & 0x0F).astype(np.int8)
        hi = (packed >> 4).astype(np.int8)
        q = np.empty(n_blocks * BLOCK, np.float32)
        q[0::2] = np.where(lo > 7, lo - 16, lo)   # sign-extend 4-bit values
        q[1::2] = np.where(hi > 7, hi - 16, hi)
    out = (q.reshape(n_blocks, BLOCK) * scale[:, None]).ravel()[:size]
    return out.reshape(entry["shape"])


def save_rcn(model, tokenizer, path: str, quant: str = "rq8", cycles: Optional[int] = None) -> Dict[str, Any]:
    """Write ``model`` (+ tokenizer) to ``path``. ``cycles`` bakes a binding-cycle profile."""
    if quant not in QUANTS:
        raise ValueError(f"quant must be one of {QUANTS}")
    cfg = model.arc1_config if cycles is None else model.arc1_config.with_cycles(cycles)
    blobs, directory, pos = [], [], 0

    def put(raw: bytes) -> int:
        nonlocal pos
        start = _align(pos)
        blobs.append(b"\x00" * (start - pos) + raw)
        pos = start + len(raw)
        return start

    for w in model.weights:
        arr = np.asarray(w.numpy())
        entry: Dict[str, Any] = {"name": w.path, "shape": list(arr.shape)}
        if quant == "f16" or arr.size < MIN_QUANT_SIZE:
            entry.update(dtype="f16", offset=put(arr.astype(np.float16).tobytes()))
        else:
            q, s = _quantize(arr, 8 if quant == "rq8" else 4)
            entry.update(dtype=quant, offset=put(q), scale_offset=put(s))
        directory.append(entry)
    data = b"".join(blobs)

    tok = {"vocab_size": tokenizer.vocab_size, "merges": [list(m) for m in tokenizer.merges]} if tokenizer else None
    meta = json.dumps({
        "format": "rcn", "model": "ARC 1", "architecture": "Resonant Schema Binding",
        "config": cfg.to_dict(), "tokenizer": tok, "profile": {"quant": quant, "cycles": cfg.resolve_cycles()},
        "tensors": directory,
    }, separators=(",", ":")).encode("utf-8")
    meta_off = HEADER_SIZE
    data_off = _align(meta_off + len(meta))
    header = _HEADER.pack(
        MAGIC, VERSION, FLAG_TOKENIZER if tok else 0, _QUANT_CODE[quant], cfg.resolve_cycles(),
        cfg.d_model, cfg.num_layers, cfg.num_heads, cfg.vocab_size, cfg.seq_len, len(directory),
        meta_off, len(meta), data_off, len(data), zlib.crc32(data),
    ).ljust(HEADER_SIZE, b"\x00")
    with open(path, "wb") as f:
        f.write(header)
        f.write(meta)
        f.write(b"\x00" * (data_off - meta_off - len(meta)))
        f.write(data)
    return {"path": path, "bytes": data_off + len(data), "tensors": len(directory), "quant": quant}


def read_header(path: str) -> Dict[str, Any]:
    """Model description from the 128-byte header (weights are not read)."""
    with open(path, "rb") as f:
        raw = f.read(HEADER_SIZE)
    if len(raw) < HEADER_SIZE or raw[:4] != MAGIC:
        raise ValueError(f"{path} is not an .rcn file")
    (_, version, flags, quant, cycles, d_model, layers, heads, vocab, seq_len, n_tensors,
     meta_off, meta_len, data_off, data_len, crc) = _HEADER.unpack(raw[:_HEADER.size])
    if version != VERSION:
        raise ValueError(f"unsupported .rcn version {version}")
    return {"version": version, "has_tokenizer": bool(flags & FLAG_TOKENIZER), "quant": QUANTS[quant],
            "cycles": cycles, "d_model": d_model, "layers": layers, "heads": heads, "vocab_size": vocab,
            "seq_len": seq_len, "tensors": n_tensors, "meta_offset": meta_off, "meta_length": meta_len,
            "data_offset": data_off, "data_length": data_len, "crc32": crc}


def _read_mapped(mm, head: Dict[str, Any], verify: bool, path: str):
    """Decode tensors from the mapping. Only float32 copies escape, so the map can close."""
    view = memoryview(mm)
    data = view[head["data_offset"]:head["data_offset"] + head["data_length"]]
    try:
        meta = json.loads(bytes(view[head["meta_offset"]:head["meta_offset"] + head["meta_length"]]))
        if verify and zlib.crc32(data) != head["crc32"]:
            raise ValueError(f"{path} is corrupted (data checksum mismatch)")
        values = [_tensor(data, entry) for entry in meta["tensors"]]
    finally:
        data.release()
        view.release()
    return meta, values


def _tensor(data: memoryview, entry: Dict[str, Any]) -> np.ndarray:
    if entry["dtype"] != "f16":
        return _dequantize(data, entry).astype(np.float32)
    size = int(np.prod(entry["shape"])) if entry["shape"] else 1
    return np.frombuffer(data, np.float16, size, entry["offset"]).reshape(entry["shape"]).astype(np.float32)


def load_rcn(path: str, verify: bool = True):
    """Map ``path`` and build a ready ``Arc1Model``. Returns ``(model, tokenizer, header)``."""
    from .arc1 import Arc1Config, Arc1Model
    from .tokenization import BytePairTokenizer

    head = read_header(path)
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        meta, values = _read_mapped(mm, head, verify, path)
    model = Arc1Model(Arc1Config.from_dict(meta["config"])).build_model()
    if len(model.weights) != len(values):
        raise ValueError("tensor count does not match the architecture")
    for w, entry in zip(model.weights, meta["tensors"]):
        if list(w.shape) != entry["shape"]:
            raise ValueError(f"shape mismatch for {entry['name']}")
    model.set_weights(values)
    tok = meta.get("tokenizer")
    tokenizer = BytePairTokenizer(tok["vocab_size"], [tuple(m) for m in tok["merges"]]) if tok else None
    return model, tokenizer, head


if __name__ == "__main__":
    import sys

    print(json.dumps(read_header(sys.argv[1]), indent=2))
