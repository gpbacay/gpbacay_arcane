"""Where is the training step time going?"""
import gc
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tensorflow as tf

from gpbacay_arcane.distillation import ArcaneDistiller
from gpbacay_arcane.language_model import ArcaneSLMConfig, ArcaneSmallLanguageModel

SEQ, BATCH, TOPK, VOCAB = 256, 8, 32, 8000
rng = np.random.default_rng(0)
inputs = tf.constant(rng.integers(4, VOCAB, size=(BATCH, SEQ)), dtype=tf.int32)
labels = tf.constant(rng.integers(4, VOCAB, size=(BATCH, SEQ)), dtype=tf.int32)
t_ids = tf.constant(rng.integers(4, VOCAB, size=(BATCH, SEQ, TOPK)), dtype=tf.int32)
t_vals = tf.constant(rng.normal(0, 1, size=(BATCH, SEQ, TOPK)), dtype=tf.float32)
mask = tf.ones((BATCH, SEQ), dtype=tf.float32)


def bench(label, **over):
    cfg = ArcaneSLMConfig.from_preset("distill-small", seq_len=SEQ, vocab_size=VOCAB, **over)
    model = ArcaneSmallLanguageModel(cfg)
    model.build_model()
    d = ArcaneDistiller(model, peak_lr=1e-3, warmup_steps=10, total_steps=100)
    d.train_step(inputs, labels, t_ids, t_vals, mask)  # trace
    n = 2
    t0 = time.time()
    for _ in range(n):
        d.train_step(inputs, labels, t_ids, t_vals, mask)
    dt = (time.time() - t0) / n
    params = int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
    print(f"{label:<34} {dt:6.2f} s/step  ({1/dt:5.3f} steps/s)  {params:,} params", flush=True)
    del d, model
    gc.collect()
    return dt


print(
    f"threads: intra={tf.config.threading.get_intra_op_parallelism_threads()} "
    f"inter={tf.config.threading.get_inter_op_parallelism_threads()}",
    flush=True,
)
bench("chunk=256 (default / 1 chunk)", chunk_size=256)
bench("chunk=64 (old scan)", chunk_size=64)
