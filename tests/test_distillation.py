import numpy as np
import pytest
import tensorflow as tf

from gpbacay_arcane.distillation import (
    ArcaneDistiller,
    WarmupCosine,
    distillation_loss,
    masked_cross_entropy,
    read_distill_dataset,
    topk_kd_loss,
    write_shard,
)
from gpbacay_arcane.language_model import ArcaneSLMConfig, ArcaneSmallLanguageModel
from gpbacay_arcane.mechanisms import (
    CausalLinearSelfAttention,
    CausalSoftmaxSelfAttention,
    RMSNorm,
    apply_rope,
    build_rope_cache,
)


# --------------------------------------------------------------------------
# Chunked linear attention
# --------------------------------------------------------------------------
def _naive_linear_attention(layer, x):
    """The original cumsum formulation, kept as the reference implementation."""
    b = tf.shape(x)[0]
    q = layer._split_heads(tf.matmul(x, layer.query_weight) + layer.query_bias, b)
    k = layer._split_heads(tf.matmul(x, layer.key_weight) + layer.key_bias, b)
    v = layer._split_heads(tf.matmul(x, layer.value_weight) + layer.value_bias, b)
    q = tf.nn.elu(q) + 1.0
    k = tf.nn.elu(k) + 1.0
    kv = tf.expand_dims(k, -1) * tf.expand_dims(v, -2)
    num = tf.einsum("bhsd,bhsde->bhse", q, tf.cumsum(kv, axis=2))
    den = tf.einsum("bhsd,bhsd->bhs", q, tf.cumsum(k, axis=2))
    return num / (tf.expand_dims(den, -1) + layer.eps)


def _features(layer, x, batch):
    q = tf.nn.elu(layer._split_heads(tf.matmul(x, layer.query_weight) + layer.query_bias, batch)) + 1.0
    k = tf.nn.elu(layer._split_heads(tf.matmul(x, layer.key_weight) + layer.key_bias, batch)) + 1.0
    v = layer._split_heads(tf.matmul(x, layer.value_weight) + layer.value_bias, batch)
    return q, k, v


@pytest.mark.parametrize("seq_len,chunk", [(6, 64), (64, 64), (100, 16), (37, 8), (128, 32)])
def test_chunked_matches_cumsum_reference(seq_len, chunk):
    """Chunking is an optimisation, not a change of semantics."""
    layer = CausalLinearSelfAttention(
        d_model=32, num_heads=4, dropout_rate=0.0,
        use_semantic_reweighting=False, chunk_size=chunk,
    )
    x = tf.random.normal((3, seq_len, 32), seed=0)
    layer(x, training=False)
    q, k, v = _features(layer, x, 3)
    got = layer._chunked_attention(q, k, v, seq_len)
    np.testing.assert_allclose(
        _naive_linear_attention(layer, x).numpy(), got.numpy(), atol=2e-5
    )


def test_single_chunk_fast_path_matches_scan():
    """The CPU fast path must match a one-iteration scan, including decay."""
    x = tf.random.normal((2, 32, 32), seed=3)
    layer = CausalLinearSelfAttention(
        d_model=32, num_heads=4, dropout_rate=0.0,
        use_semantic_reweighting=False, chunk_size=32, use_decay=True,
    )
    layer(x, training=False)
    q, k, v = _features(layer, x, 2)
    fast = layer._single_chunk_attention(q, k, v, 32)
    scanned = layer._chunked_attention(q, k, v, 32)
    np.testing.assert_allclose(fast.numpy(), scanned.numpy(), atol=2e-5)


def test_chunk_size_does_not_change_output():
    x = tf.random.normal((2, 48, 32), seed=1)
    outs = []
    for chunk in (8, 16, 48):
        layer = CausalLinearSelfAttention(
            d_model=32, num_heads=4, dropout_rate=0.0,
            use_semantic_reweighting=False, chunk_size=chunk,
        )
        layer(x, training=False)
        q, k, v = _features(layer, x, 2)
        # Same glorot seed is not guaranteed across layers, so compare against
        # the reference for each chunk size instead of against each other.
        outs.append(
            float(tf.reduce_max(tf.abs(
                _naive_linear_attention(layer, x) - layer._chunked_attention(q, k, v, 48)
            )))
        )
    assert max(outs) < 2e-5


def test_decay_is_causal_and_changes_output():
    layer = CausalLinearSelfAttention(
        d_model=16, num_heads=2, dropout_rate=0.0,
        use_semantic_reweighting=False, chunk_size=8, use_decay=True,
    )
    x = tf.random.normal((2, 24, 16), seed=2)
    y = layer(x, training=False)
    x_future = tf.concat([x[:, :10, :], x[:, 10:, :] + 4.0], axis=1)
    np.testing.assert_allclose(
        y[:, :10, :].numpy(), layer(x_future, training=False)[:, :10, :].numpy(), atol=1e-5
    )
    # A strong forget factor must actually alter the result.
    layer.decay_logit.assign(tf.fill(layer.decay_logit.shape, -2.0))
    assert float(tf.reduce_max(tf.abs(layer(x, training=False) - y))) > 1e-4


# --------------------------------------------------------------------------
# RoPE / softmax attention / RMSNorm
# --------------------------------------------------------------------------
def test_rope_preserves_relative_geometry():
    """RoPE must make the q.k score depend on offset, not absolute position."""
    cos, sin = build_rope_cache(64, 16)
    q = tf.random.normal((1, 1, 1, 16), seed=3)
    k = tf.random.normal((1, 1, 1, 16), seed=4)

    def score(i, j):
        qi = apply_rope(tf.tile(q, [1, 1, i + 1, 1]), cos, sin)[:, :, i, :]
        kj = apply_rope(tf.tile(k, [1, 1, j + 1, 1]), cos, sin)[:, :, j, :]
        return float(tf.reduce_sum(qi * kj))

    assert score(10, 5) == pytest.approx(score(30, 25), abs=1e-4)
    assert score(10, 5) != pytest.approx(score(10, 9), abs=1e-3)


def test_softmax_attention_is_causal():
    layer = CausalSoftmaxSelfAttention(d_model=16, num_heads=2, dropout_rate=0.0)
    x = tf.random.normal((2, 12, 16), seed=5)
    y = layer(x, training=False)
    x_future = tf.concat([x[:, :6, :], x[:, 6:, :] + 7.0], axis=1)
    np.testing.assert_allclose(
        y[:, :6, :].numpy(), layer(x_future, training=False)[:, :6, :].numpy(), atol=1e-5
    )


def test_rmsnorm_normalises_scale():
    norm = RMSNorm()
    x = tf.random.normal((2, 5, 32), seed=6) * 17.0
    y = norm(x)
    rms = tf.sqrt(tf.reduce_mean(tf.square(y), axis=-1))
    np.testing.assert_allclose(rms.numpy(), np.ones_like(rms.numpy()), atol=1e-3)


# --------------------------------------------------------------------------
# Hybrid student
# --------------------------------------------------------------------------
def _small_distill_config(**overrides):
    params = dict(
        vocab_size=64, d_model=32, num_layers=4, num_heads=4, ffn_mult=4,
        seq_len=16, dropout_rate=0.0, chunk_size=8, softmax_every=2,
        use_rope=True, use_decay=True, reweight_centered=True,
        gate_normalize=True, ffn_out_activation=None, norm_type="rms",
    )
    params.update(overrides)
    return ArcaneSLMConfig(**params)


def test_attention_schedule_places_softmax_layers():
    cfg = ArcaneSLMConfig.from_preset("distill")
    kinds = cfg.attention_types()
    assert len(kinds) == 12
    assert kinds == ["linear", "linear", "linear", "softmax"] * 3
    assert ArcaneSLMConfig.from_preset("100m").attention_types() == ["linear"] * 12


def test_parameter_estimate_matches_built_model():
    cfg = _small_distill_config()
    model = ArcaneSmallLanguageModel(cfg)
    model.build_model()
    actual = int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
    assert cfg.estimate_trainable_parameters() == actual


def test_hybrid_model_is_causal_and_fully_differentiable():
    cfg = _small_distill_config()
    model = ArcaneSmallLanguageModel(cfg)
    model.build_model()
    a = tf.constant([[5, 9, 13, 2, 7, 3, 11, 4]], dtype=tf.int32)
    b = tf.constant([[5, 9, 13, 2, 60, 60, 60, 60]], dtype=tf.int32)
    np.testing.assert_allclose(
        model(a, training=False)[:, :4].numpy(),
        model(b, training=False)[:, :4].numpy(),
        atol=1e-5,
    )
    with tf.GradientTape() as tape:
        logits = model(a, training=True)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(a, logits, from_logits=True)
        )
    grads = tape.gradient(loss, model.trainable_weights)
    assert all(g is not None for g in grads), "some weights receive no gradient"


def test_rope_model_skips_additive_positional_encoding():
    """Applying both RoPE and the sinusoidal table would double-count position."""
    assert ArcaneSmallLanguageModel(_small_distill_config()).positional_encoding is None
    assert ArcaneSmallLanguageModel(
        _small_distill_config(use_rope=False)
    ).positional_encoding is not None


# --------------------------------------------------------------------------
# Distillation losses
# --------------------------------------------------------------------------
def test_kd_loss_is_zero_when_student_matches_teacher():
    logits = tf.math.log(tf.constant([[[0.1, 0.2, 0.3, 0.4]]]))
    ids = tf.constant([[[0, 1, 2, 3]]], dtype=tf.int32)
    vals = tf.gather(logits, ids, batch_dims=2)
    assert float(topk_kd_loss(logits, ids, vals, temperature=1.0)) == pytest.approx(0.0, abs=1e-6)


def test_kd_loss_is_positive_on_disagreement():
    logits = tf.constant([[[3.0, 0.0, 0.0, 0.0]]])
    ids = tf.constant([[[0, 1, 2, 3]]], dtype=tf.int32)
    vals = tf.constant([[[0.0, 0.0, 0.0, 3.0]]])
    assert float(topk_kd_loss(logits, ids, vals, temperature=1.0)) > 0.5


def test_kd_loss_respects_mask():
    logits = tf.constant([[[3.0, 0.0], [3.0, 0.0]]])
    ids = tf.constant([[[0, 1], [0, 1]]], dtype=tf.int32)
    vals = tf.constant([[[0.0, 3.0], [3.0, 0.0]]])
    both = float(topk_kd_loss(logits, ids, vals, 1.0, mask=tf.constant([[1.0, 1.0]])))
    agree_only = float(topk_kd_loss(logits, ids, vals, 1.0, mask=tf.constant([[0.0, 1.0]])))
    assert agree_only < both
    assert agree_only == pytest.approx(0.0, abs=1e-6)


def test_masked_cross_entropy_ignores_masked_positions():
    logits = tf.constant([[[10.0, 0.0], [0.0, 10.0]]])
    labels = tf.constant([[0, 0]], dtype=tf.int32)  # second position is wrong
    full = float(masked_cross_entropy(logits, labels))
    first = float(masked_cross_entropy(logits, labels, mask=tf.constant([[1.0, 0.0]])))
    assert first < full
    assert first == pytest.approx(0.0, abs=1e-4)


def test_distillation_loss_blends_terms():
    logits = tf.random.normal((2, 3, 8), seed=7)
    labels = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
    ids = tf.tile(tf.constant([[[0, 1, 2, 3]]], dtype=tf.int32), [2, 3, 1])
    vals = tf.random.normal((2, 3, 4), seed=8)
    total, ce, kd = distillation_loss(logits, labels, ids, vals, alpha=0.25, temperature=2.0)
    np.testing.assert_allclose(
        float(total), 0.25 * float(ce) + 0.75 * 4.0 * float(kd), rtol=1e-5
    )


def test_warmup_cosine_shape():
    sched = WarmupCosine(peak_lr=1e-3, warmup_steps=100, total_steps=1000)
    assert float(sched(0)) == pytest.approx(0.0, abs=1e-9)
    assert float(sched(100)) == pytest.approx(1e-3, rel=1e-5)
    assert float(sched(50)) == pytest.approx(5e-4, rel=1e-5)
    assert float(sched(1000)) == pytest.approx(0.0, abs=1e-6)
    assert float(sched(2000)) == pytest.approx(0.0, abs=1e-6)  # clamped past the end


# --------------------------------------------------------------------------
# Shard round-trip + a real training step
# --------------------------------------------------------------------------
def test_shard_roundtrip_and_training_step_reduces_loss(tmp_path):
    seq_len, top_k, vocab, n = 16, 8, 64, 8
    rng = np.random.default_rng(0)
    inputs = rng.integers(4, vocab, size=(n, seq_len)).astype(np.int32)
    labels = rng.integers(4, vocab, size=(n, seq_len)).astype(np.int32)
    t_ids = np.stack([
        rng.permutation(vocab)[:top_k].astype(np.int32)
        for _ in range(n * seq_len)
    ]).reshape(n, seq_len, top_k)
    # Put the teacher's mass on the true label so the KD signal agrees with CE.
    t_ids[:, :, 0] = labels
    t_vals = rng.normal(0, 1, size=(n, seq_len, top_k)).astype(np.float16)
    t_vals[:, :, 0] = 8.0

    path = str(tmp_path / "shard_00000.tfrecord")
    assert write_shard(path, inputs, labels, t_ids, t_vals) == n

    ds = read_distill_dataset(str(tmp_path / "*.tfrecord"), seq_len, top_k,
                              batch_size=4, shuffle=False)
    batch = next(iter(ds))
    got_inputs, got_labels, got_ids, got_vals, mask = batch
    assert got_inputs.shape == (4, seq_len)
    assert got_ids.shape == (4, seq_len, top_k)
    np.testing.assert_array_equal(got_inputs.numpy(), inputs[:4])
    np.testing.assert_array_equal(got_ids.numpy(), t_ids[:4])
    np.testing.assert_allclose(got_vals.numpy(), t_vals[:4].astype(np.float32), atol=1e-2)
    assert float(tf.reduce_min(mask)) == 1.0  # no pad ids in this fixture

    student = ArcaneSmallLanguageModel(_small_distill_config(vocab_size=vocab, seq_len=seq_len))
    student.build_model()
    distiller = ArcaneDistiller(student, peak_lr=3e-3, warmup_steps=2,
                                total_steps=40, alpha=0.5, temperature=2.0)
    first = float(distiller.train_step(*batch)[0])
    for _ in range(25):
        distiller.train_step(*batch)
    last = float(distiller.train_step(*batch)[0])
    assert last < first, f"distillation loss did not decrease: {first:.4f} -> {last:.4f}"

    metrics = distiller.evaluate(ds)
    assert set(metrics) == {"loss", "ce", "kd", "ppl"}
    assert np.isfinite(metrics["loss"]) and metrics["ppl"] > 0
