"""Knowledge distillation from a softmax teacher into an ARCANE student.

The teacher (Qwen2.5-0.5B) runs in PyTorch, the student in TensorFlow, so the
two never share a graph. Teacher predictions are dumped offline as **sparse
top-k logits** in student id space and replayed from TFRecord during training.
Dense logits are not an option: 151,936 floats per position is terabytes for any
real corpus, whereas top-64 over a trimmed 32k vocabulary is a few GB.

Loss is the usual combination::

    L = alpha * CE(student, hard labels)
      + (1 - alpha) * T^2 * KL(teacher || student)

with both distributions restricted to the teacher's top-k support, which is
where essentially all of the teacher's probability mass lives anyway.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from .tokenization import PAD_ID


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------
def topk_kd_loss(
    student_logits: tf.Tensor,
    teacher_ids: tf.Tensor,
    teacher_values: tf.Tensor,
    temperature: float = 2.0,
    mask: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """KL(teacher || student) over the teacher's top-k support.

    Args:
        student_logits: ``(B, T, V)`` full student logits.
        teacher_ids: ``(B, T, K)`` student-space ids of the teacher's top-k.
        teacher_values: ``(B, T, K)`` teacher logits at those ids.
        temperature: softening temperature applied to both sides.
        mask: ``(B, T)`` 1.0 for positions to score, 0.0 to ignore.

    Both sides are renormalised over the same K symbols, so the result is a
    proper KL between two distributions on a shared support.
    """
    temperature = tf.cast(temperature, student_logits.dtype)
    teacher_values = tf.cast(teacher_values, student_logits.dtype)
    student_at_topk = tf.gather(student_logits, teacher_ids, batch_dims=2)

    teacher_logprobs = tf.nn.log_softmax(teacher_values / temperature, axis=-1)
    student_logprobs = tf.nn.log_softmax(student_at_topk / temperature, axis=-1)
    teacher_probs = tf.exp(teacher_logprobs)

    per_token = tf.reduce_sum(teacher_probs * (teacher_logprobs - student_logprobs), axis=-1)
    if mask is None:
        return tf.reduce_mean(per_token)
    mask = tf.cast(mask, per_token.dtype)
    return tf.reduce_sum(per_token * mask) / (tf.reduce_sum(mask) + 1e-8)


def masked_cross_entropy(
    student_logits: tf.Tensor,
    labels: tf.Tensor,
    mask: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Next-token cross entropy, ignoring masked positions."""
    per_token = tf.keras.losses.sparse_categorical_crossentropy(
        labels, student_logits, from_logits=True
    )
    if mask is None:
        return tf.reduce_mean(per_token)
    mask = tf.cast(mask, per_token.dtype)
    return tf.reduce_sum(per_token * mask) / (tf.reduce_sum(mask) + 1e-8)


def distillation_loss(
    student_logits: tf.Tensor,
    labels: tf.Tensor,
    teacher_ids: tf.Tensor,
    teacher_values: tf.Tensor,
    alpha: float = 0.4,
    temperature: float = 2.0,
    mask: Optional[tf.Tensor] = None,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return ``(total, ce, kd)``.

    ``alpha`` is the weight on the hard-label term. The KD term is scaled by
    ``T^2`` so its gradient magnitude stays comparable to CE as temperature moves
    (Hinton et al.).
    """
    ce = masked_cross_entropy(student_logits, labels, mask)
    kd = topk_kd_loss(student_logits, teacher_ids, teacher_values, temperature, mask)
    total = alpha * ce + (1.0 - alpha) * (temperature ** 2) * kd
    return total, ce, kd


# --------------------------------------------------------------------------
# TFRecord shards
# --------------------------------------------------------------------------
def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(
    inputs: np.ndarray,
    labels: np.ndarray,
    teacher_ids: np.ndarray,
    teacher_values: np.ndarray,
) -> bytes:
    """Pack one window. Teacher values are stored fp16 to halve shard size."""
    feature = {
        "inputs": _bytes_feature(np.asarray(inputs, dtype=np.int32).tobytes()),
        "labels": _bytes_feature(np.asarray(labels, dtype=np.int32).tobytes()),
        "teacher_ids": _bytes_feature(np.asarray(teacher_ids, dtype=np.int32).tobytes()),
        "teacher_values": _bytes_feature(np.asarray(teacher_values, dtype=np.float16).tobytes()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def write_shard(
    path: str,
    inputs: np.ndarray,
    labels: np.ndarray,
    teacher_ids: np.ndarray,
    teacher_values: np.ndarray,
) -> int:
    """Write a batch of windows to one TFRecord shard. Returns the record count."""
    with tf.io.TFRecordWriter(path) as writer:
        for i in range(len(inputs)):
            writer.write(
                serialize_example(inputs[i], labels[i], teacher_ids[i], teacher_values[i])
            )
    return len(inputs)


def read_distill_dataset(
    file_pattern,
    seq_len: int,
    top_k: int,
    batch_size: int = 8,
    shuffle: bool = True,
    shuffle_buffer: int = 2048,
    pad_id: int = PAD_ID,
) -> tf.data.Dataset:
    """Replay dumped teacher predictions as ``(inputs, labels, ids, values, mask)``.

    ``file_pattern`` is a glob string or an explicit list of shard paths; the
    list form lets a caller hold shards out for validation.
    """
    feature_spec = {
        "inputs": tf.io.FixedLenFeature([], tf.string),
        "labels": tf.io.FixedLenFeature([], tf.string),
        "teacher_ids": tf.io.FixedLenFeature([], tf.string),
        "teacher_values": tf.io.FixedLenFeature([], tf.string),
    }

    def parse(record):
        ex = tf.io.parse_single_example(record, feature_spec)
        inputs = tf.reshape(tf.io.decode_raw(ex["inputs"], tf.int32), (seq_len,))
        labels = tf.reshape(tf.io.decode_raw(ex["labels"], tf.int32), (seq_len,))
        ids = tf.reshape(tf.io.decode_raw(ex["teacher_ids"], tf.int32), (seq_len, top_k))
        vals = tf.reshape(tf.io.decode_raw(ex["teacher_values"], tf.float16), (seq_len, top_k))
        mask = tf.cast(tf.not_equal(labels, pad_id), tf.float32)
        return inputs, labels, ids, tf.cast(vals, tf.float32), mask

    files = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
    ds = files.interleave(
        tf.data.TFRecordDataset,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    return ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup then cosine decay to ``min_lr``."""

    def __init__(self, peak_lr: float, warmup_steps: int, total_steps: int, min_lr: float = 0.0):
        super().__init__()
        self.peak_lr = float(peak_lr)
        self.warmup_steps = max(int(warmup_steps), 1)
        self.total_steps = max(int(total_steps), self.warmup_steps + 1)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = self.peak_lr * step / float(self.warmup_steps)
        progress = (step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
            1.0 + tf.cos(np.pi * progress)
        )
        return tf.where(step < self.warmup_steps, warm, cosine)

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }


class ArcaneDistiller:
    """Drives distillation of a softmax teacher into an ARCANE student.

    Holds the optimizer and a compiled ``tf.function`` step so the student can be
    trained from dumped teacher logits without a live PyTorch dependency.
    """

    def __init__(
        self,
        student,
        peak_lr: float = 3e-4,
        warmup_steps: int = 500,
        total_steps: int = 20_000,
        weight_decay: float = 0.01,
        alpha: float = 0.4,
        temperature: float = 2.0,
        clipnorm: float = 1.0,
        grad_accum_steps: int = 1,
    ):
        self.student = student
        self.alpha = alpha
        self.temperature = temperature
        self.grad_accum_steps = max(int(grad_accum_steps), 1)
        schedule = WarmupCosine(peak_lr, warmup_steps, total_steps)
        optimizer_cls = getattr(tf.keras.optimizers, "AdamW", None)
        if optimizer_cls is None:  # pragma: no cover - older Keras
            optimizer_cls = tf.keras.optimizers.experimental.AdamW
        self.optimizer = optimizer_cls(
            learning_rate=schedule,
            weight_decay=weight_decay,
            beta_1=0.9,
            beta_2=0.95,
            clipnorm=clipnorm,
        )
        self._accum: Optional[List[tf.Variable]] = None

    def current_lr(self) -> float:
        """Current learning rate.

        Keras versions differ here: some expose ``learning_rate`` as the schedule
        object (callable on the step), others as the already-resolved value.
        """
        lr = self.optimizer.learning_rate
        if callable(lr):
            return float(tf.convert_to_tensor(lr(self.optimizer.iterations)))
        return float(tf.convert_to_tensor(lr))

    def _ensure_accumulators(self):
        if self._accum is None:
            self._accum = [
                tf.Variable(tf.zeros_like(w), trainable=False)
                for w in self.student.trainable_weights
            ]

    @tf.function
    def train_step(self, inputs, labels, teacher_ids, teacher_values, mask):
        with tf.GradientTape() as tape:
            logits = self.student(inputs, training=True)
            total, ce, kd = distillation_loss(
                logits,
                labels,
                teacher_ids,
                teacher_values,
                alpha=self.alpha,
                temperature=self.temperature,
                mask=mask,
            )
        weights = self.student.trainable_weights
        grads = tape.gradient(total, weights)
        self.optimizer.apply_gradients(
            (g, w) for g, w in zip(grads, weights) if g is not None
        )
        return total, ce, kd

    @tf.function
    def eval_step(self, inputs, labels, teacher_ids, teacher_values, mask):
        logits = self.student(inputs, training=False)
        return distillation_loss(
            logits,
            labels,
            teacher_ids,
            teacher_values,
            alpha=self.alpha,
            temperature=self.temperature,
            mask=mask,
        )

    def evaluate(self, dataset, max_batches: Optional[int] = None) -> dict:
        """Mean total/CE/KD plus perplexity over ``dataset``."""
        totals = np.zeros(3, dtype=np.float64)
        n = 0
        for i, batch in enumerate(dataset):
            if max_batches is not None and i >= max_batches:
                break
            values = [float(v.numpy()) for v in self.eval_step(*batch)]
            totals += values
            n += 1
        if n == 0:
            return {"loss": float("nan"), "ce": float("nan"), "kd": float("nan"), "ppl": float("nan")}
        loss, ce, kd = totals / n
        return {"loss": loss, "ce": ce, "kd": kd, "ppl": float(np.exp(min(ce, 20.0)))}
