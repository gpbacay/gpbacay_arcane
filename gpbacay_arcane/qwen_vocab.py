"""Map Qwen's BPE vocabulary onto a compact ARCANE student vocabulary.

Logit-level distillation requires teacher and student to score the *same*
symbols. Adopting Qwen2.5's tokenizer wholesale would put
``151936 x 768 = 116.7M`` parameters into the embedding table alone -- more than
the rest of a 100M-scale decoder combined -- so instead we keep only the ids
that actually occur in the target corpus and remap them into a dense student
range.

Student id layout keeps ARCANE's existing special tokens so checkpoints, the
sampler and ``ArcaneSLMConfig.pad_id`` all keep working::

    0 = PAD   1 = EOS   2 = UNK   3 = BOS   4.. = kept Qwen ids

Qwen's own end-of-text token is folded onto ``EOS`` rather than occupying a
second slot.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .tokenization import BOS_ID, EOS_ID, NUM_SPECIAL, PAD_ID, UNK_ID

QWEN_MODEL_ID = "Qwen/Qwen2.5-0.5B"


def load_qwen_tokenizer(model_id: str = QWEN_MODEL_ID):
    """Load the HF tokenizer. Kept isolated so the rest of the package stays torch-free."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "transformers is required for Qwen distillation. "
            "Install with: pip install -r requirements-distill.txt"
        ) from exc
    return AutoTokenizer.from_pretrained(model_id)


class QwenVocabAdapter:
    """A dense student vocabulary carved out of Qwen's BPE.

    ``student_to_qwen[i]`` is the Qwen id behind student id ``i`` (``-1`` for the
    four special slots). ``qwen_to_student`` is the dense inverse, with ``-1``
    for every Qwen id that did not make the cut.
    """

    def __init__(
        self,
        student_to_qwen: Sequence[int],
        qwen_vocab_size: int,
        model_id: str = QWEN_MODEL_ID,
        tokenizer=None,
    ):
        self.model_id = model_id
        self.qwen_vocab_size = int(qwen_vocab_size)
        self.student_to_qwen = np.asarray(student_to_qwen, dtype=np.int64)
        self.vocab_size = int(self.student_to_qwen.size)
        self.qwen_to_student = np.full(self.qwen_vocab_size, -1, dtype=np.int64)
        for student_id, qwen_id in enumerate(self.student_to_qwen):
            if qwen_id >= 0:
                self.qwen_to_student[qwen_id] = student_id
        self._tokenizer = tokenizer

    # -- construction -------------------------------------------------------
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = load_qwen_tokenizer(self.model_id)
        return self._tokenizer

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        vocab_size: int = 32000,
        model_id: str = QWEN_MODEL_ID,
        tokenizer=None,
    ) -> "QwenVocabAdapter":
        """Keep the ``vocab_size - 4`` most frequent Qwen ids in ``texts``."""
        if vocab_size <= NUM_SPECIAL:
            raise ValueError(f"vocab_size must exceed {NUM_SPECIAL} special tokens")
        tok = tokenizer or load_qwen_tokenizer(model_id)
        qwen_eos = tok.eos_token_id
        counts: Counter = Counter()
        for text in texts:
            counts.update(tok.encode(text))
        counts.pop(qwen_eos, None)
        keep = [tid for tid, _ in counts.most_common(vocab_size - NUM_SPECIAL)]
        # Deterministic tail-fill so the table is always exactly vocab_size wide.
        if len(keep) < vocab_size - NUM_SPECIAL:
            seen = set(keep) | {qwen_eos}
            for tid in range(len(tok)):
                if len(keep) >= vocab_size - NUM_SPECIAL:
                    break
                if tid not in seen:
                    keep.append(tid)
        mapping = [-1] * NUM_SPECIAL + keep
        adapter = cls(mapping, qwen_vocab_size=len(tok), model_id=model_id, tokenizer=tok)
        # Fold Qwen's end-of-text onto ARCANE's EOS slot.
        adapter.qwen_to_student[qwen_eos] = EOS_ID
        adapter.qwen_eos_id = qwen_eos
        return adapter

    # -- token mapping ------------------------------------------------------
    def map_qwen_ids(self, qwen_ids) -> np.ndarray:
        """Vectorised Qwen -> student id mapping; unmapped ids become ``UNK``."""
        ids = np.asarray(qwen_ids, dtype=np.int64)
        out = self.qwen_to_student[np.clip(ids, 0, self.qwen_vocab_size - 1)]
        return np.where(out < 0, UNK_ID, out).astype(np.int32)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = list(self.map_qwen_ids(self.tokenizer.encode(text)))
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return [int(i) for i in ids]

    def decode(self, token_ids: Sequence[int]) -> str:
        qwen_ids = []
        for tid in token_ids:
            tid = int(tid)
            if tid in (PAD_ID, EOS_ID, UNK_ID, BOS_ID):
                continue
            if 0 <= tid < self.vocab_size:
                qwen = int(self.student_to_qwen[tid])
                if qwen >= 0:
                    qwen_ids.append(qwen)
        return self.tokenizer.decode(qwen_ids)

    def generation_ids(self) -> List[int]:
        """Student ids that decode to real text (everything but PAD/UNK/BOS)."""
        return [EOS_ID] + list(range(NUM_SPECIAL, self.vocab_size))

    # -- persistence --------------------------------------------------------
    def save(self, path: str) -> None:
        payload = {
            "model_id": self.model_id,
            "qwen_vocab_size": self.qwen_vocab_size,
            "vocab_size": self.vocab_size,
            "student_to_qwen": [int(i) for i in self.student_to_qwen],
            "qwen_eos_id": int(getattr(self, "qwen_eos_id", -1)),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str, tokenizer=None) -> "QwenVocabAdapter":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        adapter = cls(
            payload["student_to_qwen"],
            qwen_vocab_size=int(payload["qwen_vocab_size"]),
            model_id=payload.get("model_id", QWEN_MODEL_ID),
            tokenizer=tokenizer,
        )
        eos = int(payload.get("qwen_eos_id", -1))
        if eos >= 0:
            adapter.qwen_to_student[eos] = EOS_ID
            adapter.qwen_eos_id = eos
        return adapter


def project_qwen_embeddings(
    qwen_embeddings: np.ndarray,
    adapter: QwenVocabAdapter,
    d_model: int,
    seed: int = 0,
) -> np.ndarray:
    """Warm-start the student embedding from the teacher's, projected to ``d_model``.

    Qwen2.5-0.5B is 896-wide; the student is narrower. We fit a PCA basis on the
    kept rows and project, which preserves the teacher's vocabulary geometry far
    better than a random init. Special slots get small random vectors.
    """
    rng = np.random.default_rng(seed)
    kept = adapter.student_to_qwen
    teacher_dim = qwen_embeddings.shape[1]
    out = rng.normal(0.0, 0.02, size=(adapter.vocab_size, d_model)).astype(np.float32)
    valid = kept >= 0
    rows = qwen_embeddings[kept[valid]].astype(np.float32)
    if d_model >= teacher_dim:
        out[valid, :teacher_dim] = rows
        return out
    centred = rows - rows.mean(axis=0, keepdims=True)
    # Economy SVD on the kept rows gives the top-d_model principal directions.
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    # The basis has rank min(n_rows, teacher_dim), which can be short of d_model
    # for a small vocabulary; leave any remaining columns at their random init.
    rank = min(vt.shape[0], d_model)
    projected = centred @ vt[:rank].T
    scale = 0.02 / (projected.std() + 1e-8)
    block = out[valid]
    block[:, :rank] = (projected * scale).astype(np.float32)
    out[valid] = block
    return out
