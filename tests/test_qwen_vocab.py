"""Vocabulary-adapter tests.

These use a stub tokenizer rather than the real Qwen one so the suite stays
offline and fast. The stub only has to honour the three things the adapter
touches: ``encode``, ``decode``, ``eos_token_id`` and ``__len__``.
"""

import numpy as np
import pytest

from gpbacay_arcane.qwen_vocab import QwenVocabAdapter, project_qwen_embeddings
from gpbacay_arcane.tokenization import BOS_ID, EOS_ID, NUM_SPECIAL, UNK_ID


class StubTokenizer:
    """Character-code tokenizer standing in for Qwen's BPE."""

    eos_token_id = 200

    def __len__(self):
        return 256

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


@pytest.fixture
def adapter():
    corpus = ["hello world " * 20, "hello there " * 10]
    return QwenVocabAdapter.build(
        corpus, vocab_size=NUM_SPECIAL + 12, tokenizer=StubTokenizer()
    )


def test_build_reserves_special_slots(adapter):
    assert adapter.vocab_size == NUM_SPECIAL + 12
    assert list(adapter.student_to_qwen[:NUM_SPECIAL]) == [-1] * NUM_SPECIAL
    assert all(t >= 0 for t in adapter.student_to_qwen[NUM_SPECIAL:])


def test_build_keeps_most_frequent_tokens(adapter):
    # 'l' and 'o' are the most frequent letters in the fixture corpus.
    kept = set(adapter.student_to_qwen[NUM_SPECIAL:])
    assert ord("l") in kept and ord("o") in kept
    assert ord("z") not in kept  # never appears


def test_qwen_eos_folds_onto_student_eos(adapter):
    assert adapter.qwen_to_student[StubTokenizer.eos_token_id] == EOS_ID
    assert StubTokenizer.eos_token_id not in set(adapter.student_to_qwen[NUM_SPECIAL:])


def test_encode_decode_roundtrip(adapter):
    ids = adapter.encode("hello", add_bos=True, add_eos=True)
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID
    assert adapter.decode(ids) == "hello"


def test_unmapped_tokens_become_unk(adapter):
    ids = adapter.encode("z")
    assert ids == [UNK_ID]
    assert adapter.decode(ids) == ""  # specials are dropped on decode


def test_map_qwen_ids_is_vectorised(adapter):
    arr = np.array([[ord("h"), ord("z")], [ord("e"), ord("l")]])
    out = adapter.map_qwen_ids(arr)
    assert out.shape == (2, 2)
    assert out.dtype == np.int32
    assert out[0, 1] == UNK_ID
    assert out[0, 0] == adapter.qwen_to_student[ord("h")]


def test_mapping_is_a_bijection_on_kept_ids(adapter):
    for student_id in range(NUM_SPECIAL, adapter.vocab_size):
        qwen_id = adapter.student_to_qwen[student_id]
        assert adapter.qwen_to_student[qwen_id] == student_id


def test_save_load_roundtrip(adapter, tmp_path):
    path = str(tmp_path / "adapter.json")
    adapter.save(path)
    loaded = QwenVocabAdapter.load(path, tokenizer=StubTokenizer())
    np.testing.assert_array_equal(loaded.student_to_qwen, adapter.student_to_qwen)
    np.testing.assert_array_equal(loaded.qwen_to_student, adapter.qwen_to_student)
    assert loaded.encode("hello") == adapter.encode("hello")


def test_generation_ids_exclude_pad_unk_bos(adapter):
    ids = set(adapter.generation_ids())
    assert EOS_ID in ids
    assert 0 not in ids and UNK_ID not in ids and BOS_ID not in ids


def test_build_pads_vocabulary_to_requested_size():
    """A tiny corpus must still yield a full-width table."""
    adapter = QwenVocabAdapter.build(["aaa"], vocab_size=NUM_SPECIAL + 30, tokenizer=StubTokenizer())
    assert adapter.vocab_size == NUM_SPECIAL + 30
    assert len(set(adapter.student_to_qwen[NUM_SPECIAL:])) == 30  # no duplicates


def test_embedding_projection_shape_and_scale(adapter):
    teacher = np.random.default_rng(0).normal(0, 1, size=(256, 64)).astype(np.float32)
    out = project_qwen_embeddings(teacher, adapter, d_model=16)
    assert out.shape == (adapter.vocab_size, 16)
    assert np.isfinite(out).all()
    assert 0.005 < out.std() < 0.2


def test_embedding_projection_handles_wider_student(adapter):
    teacher = np.random.default_rng(1).normal(0, 1, size=(256, 8)).astype(np.float32)
    out = project_qwen_embeddings(teacher, adapter, d_model=16)
    assert out.shape == (adapter.vocab_size, 16)
    kept = adapter.student_to_qwen >= 0
    np.testing.assert_allclose(out[kept][:, :8], teacher[adapter.student_to_qwen[kept]], atol=1e-6)
