"""Byte-level BPE tokenizer for ARCANE language models.

No extra dependencies. Special ids: pad=0, eos=1, unk=2, bos=3. UTF-8 bytes
occupy ids 4..259, then learned merges fill the rest of the vocabulary.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PAD_ID = 0
EOS_ID = 1
UNK_ID = 2
BOS_ID = 3
BYTE_OFFSET = 4
NUM_SPECIAL = 4
NUM_BYTES = 256
BASE_VOCAB = NUM_SPECIAL + NUM_BYTES


class BytePairTokenizer:
    """Trainable byte-level BPE with JSON save/load."""

    def __init__(self, vocab_size: int = 32000, merges: Optional[List[Tuple[int, int]]] = None):
        if vocab_size < BASE_VOCAB:
            raise ValueError(f"vocab_size must be >= {BASE_VOCAB} (specials + bytes)")
        self.vocab_size = int(vocab_size)
        self.merges: List[Tuple[int, int]] = [tuple(m) for m in (merges or [])]
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}

    def train(self, texts: Iterable[str], max_chars: Optional[int] = 2_000_000) -> "BytePairTokenizer":
        corpus = []
        total = 0
        for text in texts:
            if max_chars is not None and total >= max_chars:
                break
            piece = text if max_chars is None else text[: max(0, max_chars - total)]
            corpus.append(list(piece.encode("utf-8")))
            total += len(piece)
        words = [[BYTE_OFFSET + b for b in seq] for seq in corpus if seq]
        next_id = BASE_VOCAB
        self.merges = []
        while next_id < self.vocab_size:
            stats = Counter()
            for word in words:
                for i in range(len(word) - 1):
                    stats[(word[i], word[i + 1])] += 1
            if not stats:
                break
            pair, count = stats.most_common(1)[0]
            if count < 2:
                break
            self.merges.append(pair)
            words = [_merge_word(word, pair, next_id) for word in words]
            next_id += 1
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}
        return self

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = [BYTE_OFFSET + b for b in text.encode("utf-8")]
        ids = _apply_merges(ids, self._ranks)
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, token_ids: Sequence[int]) -> str:
        pieces: List[int] = []
        inverse = {BASE_VOCAB + i: pair for i, pair in enumerate(self.merges)}
        for tok in token_ids:
            tok = int(tok)
            if tok in (PAD_ID, BOS_ID, EOS_ID, UNK_ID):
                continue
            pieces.extend(_expand_token(tok, inverse))
        raw = bytes(b - BYTE_OFFSET for b in pieces if BYTE_OFFSET <= b < BYTE_OFFSET + NUM_BYTES)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        return "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")

    def generation_ids(self, printable_only: bool = False) -> List[int]:
        """Token ids that decode to real text. Used to mask LM-head sampling."""
        ids = [EOS_ID]
        if not printable_only:
            ids.extend(range(BYTE_OFFSET, BASE_VOCAB + len(self.merges)))
            return ids
        printable_bytes = set(range(32, 127)) | {9, 10}
        ids.extend(BYTE_OFFSET + b for b in sorted(printable_bytes))
        inverse = {BASE_VOCAB + i: pair for i, pair in enumerate(self.merges)}
        for merge_id in range(BASE_VOCAB, BASE_VOCAB + len(self.merges)):
            pieces = _expand_token(merge_id, inverse)
            if not pieces:
                continue
            raw = [b - BYTE_OFFSET for b in pieces]
            if all(b in printable_bytes for b in raw):
                ids.append(merge_id)
        return ids

    def save(self, path: str) -> None:
        payload = {
            "vocab_size": self.vocab_size,
            "merges": [list(pair) for pair in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "BytePairTokenizer":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        merges = [tuple(pair) for pair in payload.get("merges", [])]
        return cls(vocab_size=int(payload["vocab_size"]), merges=merges)


def _merge_word(word: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    if len(word) < 2:
        return word
    out: List[int] = []
    i = 0
    left, right = pair
    while i < len(word):
        if i < len(word) - 1 and word[i] == left and word[i + 1] == right:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def _apply_merges(ids: List[int], ranks: Dict[Tuple[int, int], int]) -> List[int]:
    if not ranks or len(ids) < 2:
        return ids
    while True:
        best = None
        best_rank = None
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            rank = ranks.get(pair)
            if rank is None:
                continue
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best = i
        if best is None:
            return ids
        new_id = BASE_VOCAB + best_rank
        ids = ids[:best] + [new_id] + ids[best + 2 :]
        if len(ids) < 2:
            return ids


def _expand_token(tok: int, inverse: Dict[int, Tuple[int, int]]) -> List[int]:
    if tok < BASE_VOCAB:
        return [tok]
    pair = inverse.get(tok)
    if pair is None:
        return []
    left, right = pair
    return _expand_token(left, inverse) + _expand_token(right, inverse)
