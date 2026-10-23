"""Occipital: the visual pathway of a System 1 model's memory.

Hippocampus recalls examples by whole words (BM25 over stemmed terms), so a request that shares no word with an
example gets no vote from it: a typo ("parcle"), another word form, a compound. The brain reads differently: the
visual word form area in the occipitotemporal cortex recognizes words from their letters, so a misspelled word
is still read. Occipital adds that pathway. It indexes each remembered example by letter 3- and 4-grams of its
words, scores a request against them with BM25, and fuses those scores with Hippocampus's word scores before the
neighbors vote::

    hippocampus = Hippocampus(model, occipital=Occipital())

Every decision model behind that Hippocampus (ARC 1, any classifier, an LLM used as a router) gets the better
recall, because the model is never involved: Occipital changes which examples vote, not how the model decides.

Two settings keep the vote useful. ``sharpness`` raises the fused score to a power, so the closest examples
dominate; with letters almost every example matches a little, and a flat vote dilutes the model's share.
``min_coverage`` keeps abstention: an example matched by letters alone counts only if it covers that share of
the request's letter evidence (idf-weighted), so a request unlike anything stored still gets no neighbors.
Both were chosen on CLINC150's validation split (``examples/benchmark_hippocampus.py --occipital``).

Port: ``node/occipital.mjs`` (same scores, plus memories stored as QR-code video).
"""

from __future__ import annotations

import math
import re
from typing import Callable, Dict, Optional

_WORD = re.compile(r"[^\W_]+")  # letters and digits, like JS [\p{L}\p{N}]+
K1, B = 1.2, 0.75


def grams(text: str) -> Dict[str, int]:
    """Letter 3- and 4-grams of each word, padded with spaces so word starts and ends count."""
    out: Dict[str, int] = {}
    for w in _WORD.findall(text.lower()):
        s = f" {w} "
        for n in (3, 4):
            for i in range(len(s) - n + 1):
                out[s[i:i + n]] = out.get(s[i:i + n], 0) + 1
    return out


class Occipital:
    """Letter-gram index over one Hippocampus's examples (give each Hippocampus its own).

    Args:
        weight: share of the fused score from letters; the rest is the word score.
        sharpness: power on the fused score (1 flattens the vote and dilutes the model's share).
        min_coverage: share of the request's letter evidence an example matched by letters alone must cover.
            Lower it to trade abstention for accuracy (0 = always find neighbors).
        max_df: a gram found in more than this share of the examples (and in more than one) is too common to
            say which example is meant: it still counts against coverage but scores nobody. Keeps lookups fast
            (each such gram would touch most of the memory) and kept no-tool refusals at 100%.
    """

    def __init__(self, weight: float = 0.5, sharpness: float = 2.0, min_coverage: float = 0.3, max_df: float = 0.1):
        self.weight, self.sharpness, self.min_coverage, self.max_df = weight, sharpness, min_coverage, max_df
        self._postings: Dict[str, Dict[str, int]] = {}
        self._len: Dict[str, int] = {}
        self._grams: Dict[str, list] = {}

    def add(self, key: str, text: str) -> None:
        self.remove(key)
        g = grams(text)
        for gram, c in g.items():
            self._postings.setdefault(gram, {})[key] = c
        self._len[key] = sum(g.values())
        self._grams[key] = list(g)

    def remove(self, key: str) -> None:
        for gram in self._grams.pop(key, ()):
            post = self._postings[gram]
            post.pop(key, None)
            if not post:
                del self._postings[gram]
        self._len.pop(key, None)

    def fuse(self, text: str, lexical: Dict[str, float],
             allowed: Optional[Callable[[str], bool]] = None) -> Dict[str, float]:
        """``lexical`` (key -> word score, top normalized to 1) fused with letter scores for ``text``, over the
        keys ``allowed`` accepts: key -> fused score."""
        n = len(self._len)
        avg = sum(self._len.values()) / max(1, n)
        scores: Dict[str, float] = {}
        seen: Dict[str, float] = {}
        total = 0.0
        for gram in grams(text):
            post = self._postings.get(gram)
            df = len(post) if post else 0
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            total += idf  # letters no example has count too: a request made of them is unlike the memory
            if not post or df > max(1, self.max_df * n):
                continue
            for key, tf in post.items():
                if allowed and not allowed(key):
                    continue
                scores[key] = scores.get(key, 0.0) + idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * self._len[key] / avg))
                seen[key] = seen.get(key, 0.0) + idf
        scores = {k: s for k, s in scores.items() if k in lexical or seen[k] >= self.min_coverage * total}
        top = max(scores.values(), default=0.0) or 1.0
        out = {k: (1 - self.weight) * s for k, s in lexical.items()}
        for k, s in scores.items():
            out[k] = out.get(k, 0.0) + self.weight * s / top
        return {k: s ** self.sharpness for k, s in out.items()}
