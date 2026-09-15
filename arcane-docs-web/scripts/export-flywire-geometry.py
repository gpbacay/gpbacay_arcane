#!/usr/bin/env python3
"""Download FlyWire FAFB v783 skeletons + the FLYWIRE neuropil mesh.

Skeletons: https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_783/<root_id>
Mesh: navis-flybrains FLYWIRE.ply (same nanometer space as the skeletons)

No CAVE token required.
"""
from __future__ import annotations

import json
import struct
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CIRCUIT = ROOT / "src" / "data" / "flywire_escape_circuit.json"
CACHE = ROOT / ".cache" / "flywire-skeletons"
OUT_BIN = ROOT / "public" / "data" / "flywire-geometry.bin"
OUT_META = ROOT / "src" / "data" / "flywire-geometry-meta.json"

SKELETON_BASE = "https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_783/"
MESH_URL = "https://raw.githubusercontent.com/navis-org/navis-flybrains/main/flybrains/meshes/FLYWIRE.ply"
TOLERANCE_NM = 800.0
DISPLAY_SPAN = 3.35


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "arcane-flywire-geometry"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=90) as res:
                return res.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError(url)


def decode_skeleton(data: bytes):
    if len(data) < 8:
        raise ValueError("truncated skeleton")
    vertices, edges = struct.unpack_from("<II", data)
    if vertices == 0 or len(data) != 8 + 16 * vertices + 8 * edges:
        raise ValueError("unexpected skeleton byte length")
    points = np.frombuffer(data, dtype="<f4", count=vertices * 3, offset=8).reshape(-1, 3).astype(np.float64)
    links = np.frombuffer(data, dtype="<u4", count=edges * 2, offset=8 + vertices * 12).reshape(-1, 2)
    radii = np.frombuffer(data, dtype="<f4", count=vertices, offset=8 + vertices * 12 + edges * 8).astype(np.float64)
    if not np.isfinite(points).all() or (edges and int(links.max()) >= vertices):
        raise ValueError("invalid skeleton")
    return points, links, radii


def simplify_path(path, points, tolerance):
    if len(path) <= 2:
        return path
    selected = {0, len(path) - 1}
    pending = [(0, len(path) - 1)]
    p = points[path]
    while pending:
        start, end = pending.pop()
        if end - start < 2:
            continue
        vector = p[end] - p[start]
        norm = float(vector @ vector)
        interior = p[start + 1 : end]
        fraction = np.clip((interior - p[start]) @ vector / max(norm, 1e-12), 0, 1)
        distances = np.linalg.norm(interior - (p[start] + fraction[:, None] * vector), axis=1)
        index = int(np.argmax(distances))
        if distances[index] > tolerance:
            split = start + 1 + index
            selected.add(split)
            pending.extend([(start, split), (split, end)])
    return [path[i] for i in sorted(selected)]


def simplify_skeleton(points, links, tolerance):
    adjacent = [[] for _ in range(len(points))]
    for a, b in links:
        a, b = int(a), int(b)
        adjacent[a].append(b)
        adjacent[b].append(a)
    seen = set()
    result = []
    starts = [i for i, nbrs in enumerate(adjacent) if len(nbrs) != 2]
    starts += [i for i, nbrs in enumerate(adjacent) if len(nbrs) == 2]
    for start in starts:
        for neighbor in adjacent[start]:
            edge = tuple(sorted((start, neighbor)))
            if edge in seen:
                continue
            seen.add(edge)
            path = [start, neighbor]
            while len(adjacent[path[-1]]) == 2:
                nxt = next(n for n in adjacent[path[-1]] if n != path[-2])
                edge = tuple(sorted((path[-1], nxt)))
                if edge in seen:
                    break
                seen.add(edge)
                path.append(nxt)
            simple = simplify_path(path, points, tolerance)
            result.extend(zip(simple[:-1], simple[1:]))
    return np.asarray(result, dtype=np.int32).reshape(-1, 2)


def parse_ply(data: bytes):
    header_end = data.find(b"end_header\n")
    if header_end < 0:
        raise ValueError("PLY header")
    header = data[:header_end].decode("ascii", "replace")
    body = data[header_end + len(b"end_header\n") :]
    n_vert = n_face = 0
    face_index = "i"
    for line in header.splitlines():
        if line.startswith("element vertex"):
            n_vert = int(line.split()[-1])
        elif line.startswith("element face"):
            n_face = int(line.split()[-1])
        elif "vertex_indices" in line and "uint" in line:
            face_index = "I"
    verts = np.frombuffer(body, dtype="<f4", count=n_vert * 3).reshape(n_vert, 3).astype(np.float64)
    faces = []
    off = n_vert * 12
    idx_dtype = np.dtype("<u4") if face_index == "I" else np.dtype("<i4")
    for _ in range(n_face):
        count = body[off]
        off += 1
        idx = np.frombuffer(body, dtype=idx_dtype, count=count, offset=off)
        off += count * 4
        faces.append(idx.astype(np.int32))
    return verts, faces


def mesh_edges(faces):
    edges = set()
    for face in faces:
        n = len(face)
        for i in range(n):
            a, b = int(face[i]), int(face[(i + 1) % n])
            edges.add((a, b) if a < b else (b, a))
    return np.asarray(list(edges), dtype=np.int32)


def region_at(p: np.ndarray) -> int:
    x, y, _z = float(p[0]), float(p[1]), float(p[2])
    if y < -1.15:
        return 5
    if x < -0.82:
        return 0
    if x > 0.82:
        return 1
    if y > 0.52 and abs(x) < 0.55:
        return 3
    if y < -0.42 and abs(x) < 0.62:
        return 4
    return 2


def neuron_region(cell_type: str, layer: str, side: str, p: np.ndarray) -> int:
    if float(p[1]) < -1.05 and layer != "sensory":
        return 5
    if cell_type == "DNp01":
        return 6
    if layer == "sensory":
        return 0 if side == "left" else 1
    return 7


def main() -> None:
    circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
    neurons = circuit["neurons"]
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT_BIN.parent.mkdir(parents=True, exist_ok=True)

    def load_one(neuron):
        root = str(neuron["id"])
        path = CACHE / f"{root}.bin"
        if not path.exists():
            path.write_bytes(fetch(SKELETON_BASE + root))
        points, links, radii = decode_skeleton(path.read_bytes())
        simplified = simplify_skeleton(points, links, TOLERANCE_NM)
        soma = points[int(np.argmax(radii))]
        return neuron, points, simplified, soma

    print(f"Fetching {len(neurons)} FlyWire v783 skeletons…", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        packed = list(pool.map(load_one, neurons))

    print("Fetching FLYWIRE neuropil mesh…", flush=True)
    mesh_verts, faces = parse_ply(fetch(MESH_URL))
    edges = mesh_edges(faces)

    all_pts = np.concatenate([mesh_verts] + [item[1] for item in packed])
    center = (all_pts.min(axis=0) + all_pts.max(axis=0)) * 0.5
    extent = float(np.max(all_pts.max(axis=0) - all_pts.min(axis=0)))
    scale = DISPLAY_SPAN / extent
    axis = np.array([1.0, -1.0, 1.0])

    def xform(p):
        return (p - center) * axis * scale

    mesh_disp = xform(mesh_verts)
    rng = np.random.default_rng(783)
    mesh_pos = []
    mesh_reg = []
    for a, b in edges:
        if rng.random() > 0.22:
            continue
        pa, pb = mesh_disp[a], mesh_disp[b]
        if np.linalg.norm(pa - pb) < 0.008:
            continue
        mesh_pos.extend([*pa, *pb])
        mesh_reg.extend([region_at(pa), region_at(pb)])

    neuron_blocks = []
    for neuron, points, simplified, soma in packed:
        disp = xform(points)
        soma_d = xform(soma)
        pos = []
        dist = []
        reg = []
        for a, b in simplified:
            pa, pb = disp[int(a)], disp[int(b)]
            pos.extend([*pa, *pb])
            dist.extend(
                [
                    float(np.linalg.norm(pa - soma_d)),
                    float(np.linalg.norm(pb - soma_d)),
                ]
            )
            ra = neuron_region(neuron["cell_type"], neuron["layer"], neuron["side"], pa)
            rb = neuron_region(neuron["cell_type"], neuron["layer"], neuron["side"], pb)
            reg.extend([ra, rb])
        neuron_blocks.append(
            {
                "id": str(neuron["id"]),
                "label": neuron.get("label", neuron["cell_type"]),
                "cell_type": neuron["cell_type"],
                "layer": neuron["layer"],
                "side": neuron["side"],
                "soma": soma_d.tolist(),
                "positions": np.asarray(pos, dtype=np.float32),
                "distances": np.asarray(dist, dtype=np.float32),
                "regions": np.asarray(reg, dtype=np.uint8),
            }
        )

    mesh_positions = np.asarray(mesh_pos, dtype=np.float32)
    mesh_regions = np.asarray(mesh_reg, dtype=np.uint8)

    chunks = [mesh_positions.tobytes(), mesh_regions.tobytes()]
    meta_neurons = []
    offset = 0
    mesh_pos_bytes = mesh_positions.nbytes
    mesh_reg_bytes = mesh_regions.nbytes
    offset = mesh_pos_bytes + mesh_reg_bytes
    for block in neuron_blocks:
        pos_off = offset
        chunks.append(block["positions"].tobytes())
        offset += block["positions"].nbytes
        dist_off = offset
        chunks.append(block["distances"].tobytes())
        offset += block["distances"].nbytes
        reg_off = offset
        chunks.append(block["regions"].tobytes())
        offset += block["regions"].nbytes
        meta_neurons.append(
            {
                "id": block["id"],
                "label": block["label"],
                "cell_type": block["cell_type"],
                "layer": block["layer"],
                "side": block["side"],
                "soma": [round(v, 5) for v in block["soma"]],
                "count": int(len(block["positions"]) // 3),
                "posOffset": pos_off,
                "distOffset": dist_off,
                "regOffset": reg_off,
            }
        )

    OUT_BIN.write_bytes(b"".join(chunks))
    meta = {
        "dataset": "flywire_fafb_public",
        "materialization": 783,
        "skeletonSource": SKELETON_BASE,
        "meshSource": MESH_URL,
        "units": "display-space from FlyWire nanometers",
        "centerNm": center.tolist(),
        "scale": scale,
        "axis": axis.tolist(),
        "bin": "/data/flywire-geometry.bin",
        "byteLength": offset,
        "mesh": {
            "count": int(len(mesh_positions) // 3),
            "posOffset": 0,
            "regOffset": mesh_pos_bytes,
        },
        "neurons": meta_neurons,
        "citation": [
            "Dorkenwald et al., Nature 2024, 10.1038/s41586-024-07558-y",
            "Schlegel et al., Nature 2024, 10.1038/s41586-024-07686-5",
        ],
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {OUT_BIN} ({OUT_BIN.stat().st_size:,} bytes), "
        f"{len(mesh_positions)//3:,} mesh verts, {len(neuron_blocks)} neurons",
        flush=True,
    )


if __name__ == "__main__":
    main()
