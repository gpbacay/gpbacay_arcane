"""ARC 1 distillation: held-out leakage filter, contrastive pair loss, and teacher distillation loss."""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpbacay_arcane.arc1 import Arc1Config, Arc1Model
from gpbacay_arcane.arc1_codec import Arc1Codec, pad_batch
from gpbacay_arcane.arc1_distill import _blocked, distill_loss, pair_loss
from gpbacay_arcane.tokenization import BASE_VOCAB, BytePairTokenizer

SMALL = dict(vocab_size=512, d_model=32, num_layers=2, num_heads=2, seq_len=64, schema_len=32,
             engram_table_size=256, engram_rows=2, binding_heads=2, binding_cycles=2)


def test_held_out_tools_are_filtered_from_the_corpus():
    for text in ("what time is it in Tokyo", "play a podcast", "turn on bluetooth", "rate it 4 stars", "good restaurants"):
        assert _blocked(text), text
    for text in ("set the timer", "check the timetable", "what's the weather", "email maria"):
        assert not _blocked(text), text


def _pair_batch(codec, rng, n=6):
    reqs = [f"request number {i} about topic {i}" for i in range(n)]
    pos = [f"tool {i}. does thing {i}" for i in range(n)]
    cols = pos + ["tool 0b. a hard negative"]
    p_index = np.arange(n, dtype=np.int32)
    hard = np.zeros((n, len(cols)), bool)
    hard[0, n] = True
    return {
        "u_ids": pad_batch([codec.utterance(t).ids for t in reqs], max_len=codec.seq_len),
        "p_ids": pad_batch([codec.schema(t) for t in cols], max_len=codec.schema_len),
        "p_index": tf.constant(p_index),
        "same": tf.constant(np.zeros((n, len(cols)), bool)),
        "hard": tf.constant(hard),
    }


def test_pair_and_distill_losses_are_finite_and_trainable():
    tf.random.set_seed(0)
    model = Arc1Model(Arc1Config(**SMALL))
    model.build_model()
    codec = Arc1Codec(BytePairTokenizer(vocab_size=BASE_VOCAB), SMALL["seq_len"], SMALL["schema_len"])
    batch = _pair_batch(codec, random.Random(0))
    projector = tf.keras.layers.Dense(16, use_bias=False)
    projector.build((None, SMALL["d_model"]))
    teacher = {"u_ids": batch["u_ids"], "s_ids": batch["p_ids"],
               "teacher": tf.random.normal([batch["u_ids"].shape[0] + batch["p_ids"].shape[0], 16])}
    variables = model.trainable_variables + projector.trainable_variables
    opt = tf.keras.optimizers.Adam(1e-2)
    first = last = None
    for _ in range(25):
        with tf.GradientTape() as tape:
            loss = pair_loss(model, batch) + distill_loss(model, projector, teacher)
        grads = tape.gradient(loss, variables)
        opt.apply_gradients([(g, v) for g, v in zip(grads, variables) if g is not None])
        first = float(loss) if first is None else first
        last = float(loss)
    assert np.isfinite(first) and np.isfinite(last)
    assert last < first  # the same batch is memorised, so the contrastive + distill loss must fall
