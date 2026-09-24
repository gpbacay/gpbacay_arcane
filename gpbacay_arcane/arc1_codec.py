"""Sequence formats shared by ARC 1 training and inference.

Every decision is one short sequence scored by the model; all candidates for a
request are padded into one batch and read in a single forward pass (Laya-style
batched option scoring). Formats:

* tool   ``TOOL name: description / USER: text / APPLIES?``       → noul
* arg    ``ARG p (type): desc / IN tool / USER: text / COPY: text / VALUE?``
         → span over the COPY tokens, noul for presence (optional params) or
           for the value itself (boolean params)
* enum   ``ARG p: desc / IN tool / USER: text / VALUE = option``   → choice
* embed  ``TEXT: text / COPY: text``                               → mean pool over COPY

The COPY ("echo") repeat lets a causal model see the whole utterance before
the tokens it points at — a causal stand-in for Laya's bidirectional encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .tokenization import BOS_ID, PAD_ID, BytePairTokenizer

SPAN_TYPES = ("string", "integer", "number")


@dataclass
class EncodedSeq:
    ids: List[int]
    span_lo: int = -1  # first COPY token index (inclusive)
    span_hi: int = -1  # last COPY token index + 1
    offsets: Optional[List[Tuple[int, int]]] = None  # byte ranges of COPY tokens in user text
    user_bytes: bytes = b""


class Arc1Codec:
    def __init__(self, tokenizer: BytePairTokenizer, seq_len: int):
        self.tok = tokenizer
        self.seq_len = int(seq_len)

    # ---------------------------------------------------------------- helpers
    def _ids(self, text: str) -> List[int]:
        return self.tok.encode(text)

    def _user(self, user: str, budget: int) -> Tuple[List[int], List[Tuple[int, int]], bytes]:
        ids, offsets = self.tok.encode_with_offsets(user)
        budget = max(budget, 1)
        ids, offsets = ids[:budget], offsets[:budget]
        raw = user.encode("utf-8")[: offsets[-1][1] if offsets else 0]
        return ids, offsets, raw

    def _fit(self, prefix: List[int], suffix: List[int], user: str, copies: int):
        budget = (self.seq_len - len(prefix) - len(suffix) - 1 - 8 * copies) // max(copies, 1)
        return self._user(user, budget)

    # ---------------------------------------------------------------- formats
    def tool_seq(self, name: str, description: str, user: str) -> EncodedSeq:
        prefix = self._ids(f"TOOL {name}: {description}\nUSER: ")
        suffix = self._ids("\nAPPLIES?")
        uids, _, _ = self._fit(prefix, suffix, user, 1)
        return EncodedSeq([BOS_ID] + prefix + uids + suffix)

    def arg_seq(self, tool_name: str, pname: str, ptype: str, pdesc: str, required: bool, user: str) -> EncodedSeq:
        req = "" if required else ", optional"
        prefix = self._ids(f"ARG {pname} ({ptype}{req}): {pdesc}\nIN {tool_name}\nUSER: ")
        mid = self._ids("\nCOPY: ")
        suffix = self._ids("\nVALUE?")
        uids, offsets, raw = self._fit(prefix + mid, suffix, user, 2)
        ids = [BOS_ID] + prefix + uids + mid
        lo = len(ids)
        ids = ids + uids
        hi = len(ids)
        return EncodedSeq(ids + suffix, lo, hi, offsets, raw)

    def enum_seq(self, tool_name: str, pname: str, pdesc: str, value: str, user: str) -> EncodedSeq:
        prefix = self._ids(f"ARG {pname}: {pdesc}\nIN {tool_name}\nUSER: ")
        suffix = self._ids(f"\nVALUE = {value}")
        uids, _, _ = self._fit(prefix, suffix, user, 1)
        return EncodedSeq([BOS_ID] + prefix + uids + suffix)

    def embed_seq(self, text: str) -> EncodedSeq:
        prefix = self._ids("TEXT: ")
        mid = self._ids("\nCOPY: ")
        uids, offsets, raw = self._fit(prefix + mid, [], text, 2)
        ids = [BOS_ID] + prefix + uids + mid
        lo = len(ids)
        ids = ids + uids
        return EncodedSeq(ids, lo, len(ids), offsets, raw)

    # ------------------------------------------------------------------ spans
    @staticmethod
    def char_span_to_tokens(seq: EncodedSeq, user: str, span: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Map a ``[char_start, char_end)`` span to absolute (start_tok, end_tok) inclusive."""
        if seq.offsets is None or not seq.offsets:
            return None
        b0 = len(user[: span[0]].encode("utf-8"))
        b1 = len(user[: span[1]].encode("utf-8"))
        if b1 <= b0 or b1 > seq.offsets[-1][1]:
            return None  # truncated away
        start = end = None
        for i, (a, b) in enumerate(seq.offsets):
            if start is None and a <= b0 < b:
                start = i
            if a < b1 <= b:
                end = i
                break
        if start is None or end is None or end < start:
            return None
        return seq.span_lo + start, seq.span_lo + end

    @staticmethod
    def tokens_to_text(seq: EncodedSeq, start: int, end: int) -> str:
        a = seq.offsets[start - seq.span_lo][0]
        b = seq.offsets[end - seq.span_lo][1]
        text = seq.user_bytes[a:b].decode("utf-8", errors="ignore")
        return text.strip().strip("\"'").strip(" .,!?;:")


def pad_batch(seqs: Sequence[Sequence[int]], multiple: int = 16, max_len: Optional[int] = None) -> np.ndarray:
    """Right-pad to a shared length rounded up to ``multiple`` (limits retracing)."""
    longest = max((len(s) for s in seqs), default=1)
    width = int(np.ceil(longest / multiple) * multiple)
    if max_len is not None:
        width = min(width, int(max_len))
    out = np.full((len(seqs), width), PAD_ID, dtype=np.int32)
    for i, s in enumerate(seqs):
        s = list(s)[:width]
        out[i, : len(s)] = s
    return out


def best_span(start_logits: np.ndarray, end_logits: np.ndarray, lo: int, hi: int,
              temperature: float = 1.0, max_width: int = 96) -> Tuple[int, int, float]:
    """Highest-probability (start <= end) span inside [lo, hi); returns calibrated P."""
    s = start_logits[lo:hi].astype(np.float64) / temperature
    e = end_logits[lo:hi].astype(np.float64) / temperature
    ps = np.exp(s - s.max()); ps /= ps.sum()
    pe = np.exp(e - e.max()); pe /= pe.sum()
    n = hi - lo
    joint = np.triu(np.outer(ps, pe))
    if max_width < n:
        joint = np.tril(joint, max_width)
    idx = int(np.argmax(joint))
    i, j = divmod(idx, n)
    return lo + i, lo + j, float(joint[i, j])
