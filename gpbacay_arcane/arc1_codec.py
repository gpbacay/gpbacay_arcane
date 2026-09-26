"""Text formats shared by ARC 1 training and inference.

ARC 1 reads two kinds of text:

* the **utterance** — ``[BOS] + user tokens``, perceived once per request.
  Byte offsets of every token are kept so an anchored span maps back to the
  exact characters the user typed.
* **schema texts** — one short line per tool, parameter, and enum option.
  Each is perceived once, pooled into a schema engram, and cached.

Probe roles tell the binding which readout a probe feeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .tokenization import BOS_ID, PAD_ID, BytePairTokenizer

SPAN_TYPES = ("string", "integer", "number")

ROLE_TOOL = 0      # does this tool fire for the utterance?
ROLE_SPAN = 1      # string / number argument: anchored copy (+ presence when optional)
ROLE_BOOL = 2      # boolean argument: fire = value is true
ROLE_ENUM = 3      # enum argument: select over its option probes (+ presence)
ROLE_OPTION = 4    # one enum option
NUM_ROLES = 5


def role_for(ptype: str, enum: Optional[Sequence[str]]) -> int:
    if enum:
        return ROLE_ENUM
    if (ptype or "string").lower() in ("boolean", "bool"):
        return ROLE_BOOL
    return ROLE_SPAN


def _humanize(name: str) -> str:
    return str(name).replace("_", " ").replace("-", " ").strip()


def tool_text(name: str, description: str) -> str:
    return f"{_humanize(name)}. {description}".strip()


def param_text(name: str, ptype: str, description: str, required: bool) -> str:
    req = "" if required else ", optional"
    return f"{_humanize(name)} ({ptype or 'string'}{req}). {description}".strip()


def option_text(value: str, description: Optional[str] = None) -> str:
    text = _humanize(str(value))
    return f"{text}: {description.strip()}" if description and description.strip() else text


@dataclass
class Utterance:
    ids: List[int]                       # [BOS] + tokens
    offsets: List[Tuple[int, int]]       # byte range of ids[1:] in ``raw``
    raw: bytes

    @property
    def lo(self) -> int:
        return 1

    @property
    def hi(self) -> int:
        return len(self.ids)


class Arc1Codec:
    def __init__(self, tokenizer: BytePairTokenizer, seq_len: int, schema_len: int = 64):
        self.tok = tokenizer
        self.seq_len = int(seq_len)
        self.schema_len = int(schema_len)

    def utterance(self, text: str) -> Utterance:
        ids, offsets = self.tok.encode_with_offsets(text)
        budget = max(self.seq_len - 1, 1)
        ids, offsets = ids[:budget], offsets[:budget]
        raw = text.encode("utf-8")[: offsets[-1][1] if offsets else 0]
        return Utterance([BOS_ID] + ids, offsets, raw)

    def schema(self, text: str) -> List[int]:
        return [BOS_ID] + self.tok.encode(text)[: self.schema_len - 1]

    # ------------------------------------------------------------------ spans
    @staticmethod
    def char_span_to_tokens(utt: Utterance, text: str, span: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Map a ``[char_start, char_end)`` span to (start_tok, end_tok) inclusive, in ``utt.ids``."""
        if not utt.offsets:
            return None
        b0 = len(text[: span[0]].encode("utf-8"))
        b1 = len(text[: span[1]].encode("utf-8"))
        if b1 <= b0 or b1 > utt.offsets[-1][1]:
            return None  # truncated away
        start = end = None
        for i, (a, b) in enumerate(utt.offsets):
            if start is None and a <= b0 < b:
                start = i
            if a < b1 <= b:
                end = i
                break
        if start is None or end is None or end < start:
            return None
        return utt.lo + start, utt.lo + end

    @staticmethod
    def tokens_to_text(utt: Utterance, start: int, end: int) -> str:
        a = utt.offsets[start - utt.lo][0]
        b = utt.offsets[end - utt.lo][1]
        text = utt.raw[a:b].decode("utf-8", errors="ignore")
        return text.strip().strip("\"'").strip(" .,!?;:")

    @staticmethod
    def tokens_to_char_span(utt: Utterance, start: int, end: int) -> Tuple[int, int]:
        """``[char_start, char_end)`` of the trimmed anchored surface in the original text."""
        a = utt.offsets[start - utt.lo][0]
        b = utt.offsets[end - utt.lo][1]
        piece = utt.raw[a:b].decode("utf-8", errors="ignore")
        core = piece.strip().strip("\"'").strip(" .,!?;:")
        c0 = len(utt.raw[:a].decode("utf-8", errors="ignore")) + max(piece.find(core), 0)
        return c0, c0 + len(core)


def pad_batch(seqs: Sequence[Sequence[int]], multiple: int = 8, max_len: Optional[int] = None) -> np.ndarray:
    """Right-pad to a shared length rounded up to ``multiple``."""
    longest = max((len(s) for s in seqs), default=1)
    width = max(int(np.ceil(longest / multiple) * multiple), multiple)
    if max_len is not None:
        width = min(width, int(max_len))
    out = np.full((len(seqs), width), PAD_ID, dtype=np.int32)
    for i, s in enumerate(seqs):
        s = list(s)[:width]
        out[i, : len(s)] = s
    return out


def best_span(start_logits: np.ndarray, end_logits: np.ndarray, lo: int, hi: int,
              temperature: float = 1.0, max_width: int = 64) -> Tuple[int, int, float]:
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
