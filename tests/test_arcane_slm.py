import numpy as np
import tensorflow as tf

from gpbacay_arcane.language_model import (
    ArcaneSLMConfig,
    ArcaneSmallLanguageModel,
    causal_windows,
)
from gpbacay_arcane.mechanisms import CausalLinearSelfAttention, ResonantSequenceMixer
from gpbacay_arcane.tokenization import BASE_VOCAB, BytePairTokenizer


def test_causal_linear_attention_ignores_future():
    layer = CausalLinearSelfAttention(d_model=8, num_heads=2, dropout_rate=0.0, use_semantic_reweighting=False)
    x = tf.random.normal((2, 6, 8))
    y = layer(x, training=False)
    x_future = tf.concat([x[:, :3, :], x[:, 3:, :] + 5.0], axis=1)
    y_future = layer(x_future, training=False)
    np.testing.assert_allclose(y[:, :3, :].numpy(), y_future[:, :3, :].numpy(), atol=1e-5)
    assert y.shape == (2, 6, 8)


def test_resonant_sequence_mixer_is_causal():
    layer = ResonantSequenceMixer(d_model=8, resonance_factor=0.3, resonance_cycles=4)
    x = tf.random.normal((2, 7, 8))
    y = layer(x, training=False)
    x_future = tf.concat([x[:, :4, :], x[:, 4:, :] + 3.0], axis=1)
    y_future = layer(x_future, training=False)
    np.testing.assert_allclose(y[:, :4, :].numpy(), y_future[:, :4, :].numpy(), atol=1e-5)


def test_tiny_slm_next_token_shapes_and_loss():
    model = ArcaneSmallLanguageModel.from_preset("tiny", seq_len=16)
    model.build_model()
    tokens = tf.random.uniform((2, 16), minval=0, maxval=512, dtype=tf.int32)
    logits = model(tokens, training=True)
    assert logits.shape == (2, 16, 512)
    loss = tf.keras.losses.sparse_categorical_crossentropy(tokens, logits, from_logits=True)
    assert loss.shape == (2, 16)
    assert float(tf.reduce_mean(loss).numpy()) > 0


def test_tiny_slm_generate_length():
    model = ArcaneSmallLanguageModel.from_preset("tiny", seq_len=8, dropout_rate=0.0)
    model.build_model()
    out = model.generate([4, 5, 6], max_new_tokens=5, temperature=0.0, eos_id=None)
    assert len(out) == 8


def test_tiny_slm_forward_short_sequence():
    model = ArcaneSmallLanguageModel.from_preset("tiny", seq_len=16)
    model.build_model()
    logits = model(tf.constant([[4, 5, 6]], dtype=tf.int32), training=False)
    assert logits.shape == (1, 3, 512)


def test_causal_windows_shifts_targets():
    x, y = causal_windows([1, 2, 3, 4, 5], seq_len=3)
    np.testing.assert_array_equal(x[0], [1, 2, 3])
    np.testing.assert_array_equal(y[0], [2, 3, 4])


def test_byte_pair_tokenizer_roundtrip():
    tok = BytePairTokenizer(vocab_size=BASE_VOCAB + 32)
    tok.train(["hello hello world hello world"], max_chars=None)
    text = "hello world"
    ids = tok.encode(text, add_bos=True, add_eos=True)
    assert ids[0] == 3
    assert ids[-1] == 1
    assert tok.decode(ids) == text


def test_byte_pair_tokenizer_skips_unknown_ids():
    tok = BytePairTokenizer(vocab_size=BASE_VOCAB + 32)
    assert tok.decode([4 + ord("A"), 9999, 4 + ord("B")]) == "AB"


def test_byte_pair_tokenizer_printable_generation_ids():
    tok = BytePairTokenizer(vocab_size=BASE_VOCAB)
    ids = tok.generation_ids(printable_only=True)
    assert 1 in ids
    assert (4 + ord("h")) in ids
    assert (4 + 0) not in ids
    assert (4 + 255) not in ids


def test_printable_generation_ids_include_printable_merges():
    th = (4 + ord("t"), 4 + ord("h"))
    tok = BytePairTokenizer(vocab_size=BASE_VOCAB + 1, merges=[th])
    ids = tok.generation_ids(printable_only=True)
    assert BASE_VOCAB in ids
    control = BytePairTokenizer(vocab_size=BASE_VOCAB + 1, merges=[(4, 5)])
    assert BASE_VOCAB not in control.generation_ids(printable_only=True)


def test_sample_logits_respects_allowed_ids():
    from gpbacay_arcane.language_model import _sample_logits

    logits = np.array([10.0, 0.0, 9.0], dtype=np.float32)
    picked = {_sample_logits(logits, temperature=0.0, allowed_ids=[1]) for _ in range(3)}
    assert picked == {1}


def test_100m_preset_estimate_is_near_one_hundred_million():
    cfg = ArcaneSLMConfig.from_preset("100m")
    estimate = cfg.estimate_trainable_parameters()
    assert 90_000_000 <= estimate <= 110_000_000
