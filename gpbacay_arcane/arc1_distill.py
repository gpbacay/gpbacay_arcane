"""Teacher distillation, contrastive pre-training, and replay for ARC 1.

A staged recipe adapted from Contrastive Language Models (CLM):

* **A** — contrastive request -> tool / intent pre-training (bidirectional
  InfoNCE between the pooled request embedding and the tool engram) plus
  teacher distillation.
* **B** — the same with hard negatives mined from the teacher's own
  similarities (nearest wrong tools / intents).
* **C** — the usual multi-task training (``arc1_train.compute_losses``) plus
  distillation and 40% contrastive replay, so stage A is not forgotten.

The teacher (Qwen3-Embedding-0.6B) runs offline once
(``examples/dump_arc1_teacher.py``) and only its cached vectors are used here.
The projection onto the teacher space is training-only: it is not part of
``Arc1Model``, so the exported model keeps its size and latency.

Build the corpus (main env), embed it (teacher env), then train::

    python -m gpbacay_arcane.arc1_distill --out data/arc1_teacher
    teacher-venv/python examples/dump_arc1_teacher.py --corpus data/arc1_teacher
    python examples/train_arc1.py --teacher-cache data/arc1_teacher --out-dir Models/distill
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from .arc1_codec import Arc1Codec, option_text, pad_batch, tool_text
from .arc1_data import (
    HELD_OUT_TOOLS,
    build_tool_library,
    fixed_classify_set,
    fixed_eval_set,
    fixed_extract_set,
    sample_classify_example,
    sample_tool_example,
)
from .arc1_train import BATCH_SIGNATURE, compute_losses, sample_batch

_NEG = -1e9

# ------------------------------------------------------------------ corpus
# Anything resembling a held-out tool is kept out of the corpus, so the
# unseen-tool evaluation stays zero-shot.
HELD_OUT_INTENTS = {"time", "timezone", "restaurant_reservation", "restaurant_reviews", "restaurant_suggestion"}
HELD_OUT_WORDS = re.compile(r"\b(podcasts?|bluetooth|restaurants?|movies?|films?|stars|what time|time is it|the time"
                            r"|time in|places to eat|where can i eat|star rating)\b")
# CLINC150 intents held out entirely: zero-shot classification on real paraphrases.
REAL_EVAL_INTENT_STRIDE = 5


def _norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _blocked(text: str) -> bool:
    return HELD_OUT_WORDS.search(text.lower()) is not None


def _humanize(name: str) -> str:
    return name.replace("_", " ")


def real_eval_intents(clinc_path: str) -> List[str]:
    with open(clinc_path, encoding="utf-8") as f:
        intents = sorted({lab for _, lab in json.load(f)["train"]} - HELD_OUT_INTENTS)
    return intents[::REAL_EVAL_INTENT_STRIDE]


def _eval_texts() -> set:
    """Every text the evaluation sets use (normalised), to drop from the corpus."""
    texts = [ex.user for s in ("eval", "unseen_tools") for ex in fixed_eval_set(s, 1000)]
    texts += [ex.text for s in ("eval", "unseen_tools") for ex in fixed_classify_set(s, 500)]
    texts += [ex.text for ex in fixed_extract_set(500)]
    return {_norm_text(t) for t in texts}


def build_corpus(out_dir: str, external_dir: str = "data/external", n_tool: int = 20000,
                 n_classify: int = 6000, seed: int = 0) -> Dict[str, int]:
    """Write ``texts.json`` (text, kind u|s) and ``pairs.json`` (request -> positive, group, source)."""
    rng = random.Random(seed)
    lib = build_tool_library()
    banned = _eval_texts()
    texts: Dict[str, str] = {}  # text -> "u" (utterance) | "s" (schema-like)
    pairs: List[Tuple[str, str, str, str]] = []

    def add_pair(u: str, p: str, group: str, source: str) -> None:
        if _norm_text(u) in banned or _blocked(u) or _blocked(p):
            return
        texts.setdefault(u, "u")
        texts.setdefault(p, "s")
        pairs.append((u, p, group, source))

    for _ in range(n_tool):
        ex = sample_tool_example(rng, lib, "train")
        if ex.kind != "tool":
            continue
        spec = next(s for s in ex.tools if s.name == ex.calls[0].tool)
        add_pair(ex.user, tool_text(spec.name, spec.description), spec.name, "tools")

    for _ in range(n_classify):  # distillation only: classification texts and option texts
        ex = sample_classify_example(rng, lib, "train")
        if _norm_text(ex.text) in banned or _blocked(ex.text):
            continue
        texts.setdefault(ex.text, "u")
        for lab in ex.labels:
            texts.setdefault(option_text(lab, ex.descriptions.get(lab)), "s")

    clinc = os.path.join(external_dir, "clinc150_data_full.json")
    held = set(real_eval_intents(clinc)) | HELD_OUT_INTENTS
    with open(clinc, encoding="utf-8") as f:
        data = json.load(f)
    for text, lab in data["train"] + data["val"]:
        if lab not in held:
            add_pair(text, _humanize(lab), lab, "clinc")
    with open(os.path.join(external_dir, "banking77_train.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            add_pair(row["text"], _humanize(row["category"]), row["category"], "banking")

    os.makedirs(out_dir, exist_ok=True)
    items = sorted(texts.items())
    with open(os.path.join(out_dir, "texts.json"), "w", encoding="utf-8") as f:
        json.dump({"texts": [t for t, _ in items], "kinds": [k for _, k in items]}, f)
    with open(os.path.join(out_dir, "pairs.json"), "w", encoding="utf-8") as f:
        json.dump(pairs, f)
    counts = {"texts": len(items), "pairs": len(pairs)}
    for src in ("tools", "clinc", "banking"):
        counts[f"pairs_{src}"] = sum(1 for p in pairs if p[3] == src)
    return counts


# ------------------------------------------------------------- teacher cache
class TeacherCache:
    """Teacher vectors for the corpus, plus contrastive pairs and hard negatives."""

    def __init__(self, cache_dir: str, n_hard: int = 3):
        with open(os.path.join(cache_dir, "texts.json"), encoding="utf-8") as f:
            meta = json.load(f)
        with open(os.path.join(cache_dir, "pairs.json"), encoding="utf-8") as f:
            pairs = json.load(f)
        self.texts: List[str] = meta["texts"]
        self.kinds = np.asarray([k == "u" for k in meta["kinds"]])
        self.vectors = np.load(os.path.join(cache_dir, "teacher.npy")).astype(np.float32)
        assert len(self.vectors) == len(self.texts), "teacher.npy does not match texts.json; re-run the dump"
        index = {t: i for i, t in enumerate(self.texts)}
        self.u_rows = np.flatnonzero(self.kinds).tolist()
        self.s_rows = np.flatnonzero(~self.kinds).tolist()
        self.pairs: Dict[str, List[Tuple[int, int, str]]] = {}
        for u, p, group, source in pairs:
            self.pairs.setdefault(source, []).append((index[u], index[p], group))
        self.hard = self._mine_hard(n_hard)

    def _mine_hard(self, k: int) -> Dict[int, List[int]]:
        """For every positive text: the k teacher-nearest positives of another group, same source."""
        hard: Dict[int, List[int]] = {}
        for rows in self.pairs.values():
            group_of = {p: g for _, p, g in rows}
            pos = np.asarray(sorted(group_of))
            groups = np.asarray([group_of[p] for p in pos])
            v = self.vectors[pos]
            sim = v @ v.T
            sim[groups[:, None] == groups[None, :]] = -np.inf
            nearest = np.argsort(-sim, axis=1)[:, :k]
            for i, p in enumerate(pos):
                hard[int(p)] = [int(pos[j]) for j in nearest[i]]
        return hard


# -------------------------------------------------------------------- batches
PAIR_SIGNATURE = {
    "u_ids": tf.TensorSpec([None, None], tf.int32),   # requests
    "p_ids": tf.TensorSpec([None, None], tf.int32),   # positives, then hard negatives
    "p_index": tf.TensorSpec([None], tf.int32),       # row of request i's positive in p_ids
    "same": tf.TensorSpec([None, None], tf.bool),     # (B, P) column shares request i's group (not its positive)
    "hard": tf.TensorSpec([None, None], tf.bool),     # (B, P) column is one of request i's hard negatives
}
DISTILL_SIGNATURE = {
    "u_ids": tf.TensorSpec([None, None], tf.int32),
    "s_ids": tf.TensorSpec([None, None], tf.int32),
    "teacher": tf.TensorSpec([None, None], tf.float32),  # rows: utterances first, then schema texts
}


def sample_pairs(rng: random.Random, codec: Arc1Codec, cache: TeacherCache, n: int, hard: bool) -> Dict[str, np.ndarray]:
    """One source per batch, so in-batch negatives never collide across datasets (CLINC 'weather' vs get_weather)."""
    source = rng.choice(sorted(cache.pairs))
    rows = rng.sample(cache.pairs[source], min(n, len(cache.pairs[source])))
    cols: List[int] = []
    col_of: Dict[int, int] = {}
    group_of_col: List[str] = []

    def col(p: int, group: str) -> int:
        if p not in col_of:
            col_of[p] = len(cols)
            cols.append(p)
            group_of_col.append(group)
        return col_of[p]

    p_index = [col(p, g) for _, p, g in rows]
    hard_sets = []
    for _, p, _g in rows:
        hard_sets.append({col(h, "") for h in cache.hard.get(p, [])} if hard else set())
    groups = [g for _, _, g in rows]
    same = np.array([[group_of_col[j] == groups[i] and j != p_index[i] for j in range(len(cols))]
                     for i in range(len(rows))])
    hard_m = np.array([[j in hard_sets[i] for j in range(len(cols))] for i in range(len(rows))])
    return {
        "u_ids": pad_batch([codec.utterance(cache.texts[u]).ids for u, _, _ in rows], multiple=16, max_len=codec.seq_len),
        "p_ids": pad_batch([codec.schema(cache.texts[p]) for p in cols], multiple=16, max_len=codec.schema_len),
        "p_index": np.asarray(p_index, dtype=np.int32),
        "same": same,
        "hard": hard_m,
    }


def sample_distill(rng: random.Random, codec: Arc1Codec, cache: TeacherCache, n_u: int, n_s: int) -> Dict[str, np.ndarray]:
    u = rng.sample(cache.u_rows, n_u)
    s = rng.sample(cache.s_rows, n_s)
    return {
        "u_ids": pad_batch([codec.utterance(cache.texts[i]).ids for i in u], multiple=16, max_len=codec.seq_len),
        "s_ids": pad_batch([codec.schema(cache.texts[i]) for i in s], multiple=16, max_len=codec.schema_len),
        "teacher": cache.vectors[u + s],
    }


# --------------------------------------------------------------------- losses
def _soft_ce(logits, targets):
    return tf.reduce_mean(tf.reduce_sum(-targets * tf.nn.log_softmax(logits, axis=-1), axis=-1))


def pair_loss(model, batch, temperature: float = 0.05, training: bool = True):
    """Bidirectional InfoNCE: request -> positive over in-batch positives (+ its own hard
    negatives); positive -> request over the batch. Same-group columns are masked."""
    u = model.embed_text(batch["u_ids"], training=training)                          # (B, D), L2-normed
    p = tf.nn.l2_normalize(model.schema_engrams(batch["p_ids"], training=training), axis=-1)  # (P, D)
    sim = tf.matmul(u, p, transpose_b=True) / temperature                            # (B, P)
    # Other requests' hard negatives are ordinary in-batch columns only if they are someone's positive.
    is_pos_col = tf.reduce_any(tf.equal(tf.range(tf.shape(p)[0])[None, :], batch["p_index"][:, None]), axis=0)
    keep = tf.logical_and(tf.logical_or(is_pos_col[None, :], batch["hard"]), tf.logical_not(batch["same"]))
    s2p = tf.where(keep, sim, _NEG)
    loss_s2p = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch["p_index"], logits=s2p))
    p2s = tf.transpose(tf.gather(sim, batch["p_index"], axis=1))                     # (B, B): positive i vs request j
    groups_same = tf.gather(batch["same"], batch["p_index"], axis=1)                 # request i shares group with p_j
    eye = tf.eye(tf.shape(u)[0], dtype=tf.bool)
    p2s = tf.where(tf.logical_and(tf.transpose(groups_same), tf.logical_not(eye)), _NEG, p2s)
    dup = tf.equal(batch["p_index"][:, None], batch["p_index"][None, :])            # same positive text twice
    p2s = tf.where(tf.logical_and(dup, tf.logical_not(eye)), _NEG, p2s)
    loss_p2s = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.range(tf.shape(u)[0]), logits=p2s))
    return 0.5 * (loss_s2p + loss_p2s)


def distill_loss(model, projector, batch, temperature: float = 0.05, training: bool = True):
    """Relational distillation: the student's similarity to every teacher vector in the
    batch should match the teacher's own similarity distribution (soft InfoNCE)."""
    u = model.embed_text(batch["u_ids"], training=training)
    s = model.schema_engrams(batch["s_ids"], training=training)
    student = tf.nn.l2_normalize(projector(tf.concat([u, s], axis=0)), axis=-1)
    teacher = tf.nn.l2_normalize(batch["teacher"], axis=-1)
    targets = tf.nn.softmax(tf.matmul(teacher, teacher, transpose_b=True) / temperature, axis=-1)
    logits = tf.matmul(student, teacher, transpose_b=True) / temperature
    return 0.5 * (_soft_ce(logits, targets) + _soft_ce(tf.transpose(logits), targets))


# ------------------------------------------------------------------- training
@dataclass
class DistillConfig:
    steps_a: int = 3000
    steps_b: int = 1000
    lr_a: float = 2e-3
    lr_b: float = 1e-3
    n_pairs: int = 64
    n_distill_u: int = 48
    n_distill_s: int = 16
    distill_weight: float = 1.0      # stages A/B
    distill_weight_c: float = 0.5    # stage C
    replay_weight: float = 0.5       # stage C
    replay_frac: float = 0.4         # replayed pairs, as a share of the stage C mixture
    n_hard: int = 3


def _optimizer(lr: float, steps: int, warmup: int, wd: float):
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr * 0.05, decay_steps=max(steps - warmup, 1), alpha=0.05,
        warmup_target=lr, warmup_steps=warmup)
    return tf.keras.optimizers.AdamW(learning_rate=schedule, weight_decay=wd, clipnorm=1.0)


def _log(stage: str, step: int, steps: int, running: Dict[str, float], every: int, t0: float, extra: str = "") -> None:
    avg = {k: v / every for k, v in running.items()}
    elapsed = time.time() - t0
    print(f"[arc1:{stage}] step {step}/{steps} {extra}" + " ".join(f"{k}={v:.3f}" for k, v in avg.items())
          + f" | {elapsed / step:.2f}s/step eta {elapsed / step * (steps - step) / 60:.1f}m", flush=True)


def train_distilled(model, tokenizer, tc, dc: DistillConfig, cache_dir: str, out_prefix: str,
                    start_stage: str = "A") -> Dict[str, Any]:
    """Stages A -> B -> C. ``tc`` is the usual ``TrainConfig`` (used for stage C)."""
    cfg = model.arc1_config
    codec = Arc1Codec(tokenizer, cfg.seq_len, cfg.schema_len)
    cache = TeacherCache(cache_dir, dc.n_hard)
    lib = build_tool_library()
    rng = random.Random(tc.seed)
    tf.random.set_seed(tc.seed)
    projector = tf.keras.layers.Dense(cache.vectors.shape[1], use_bias=False, name="teacher_projector")
    projector.build((None, cfg.d_model))
    print(f"[arc1] teacher cache: {len(cache.texts):,} texts ({cache.vectors.shape[1]}d), "
          + ", ".join(f"{k}={len(v):,} pairs" for k, v in sorted(cache.pairs.items())), flush=True)
    info: Dict[str, Any] = {"history": []}
    t_all = time.time()

    order = "ABC"
    if start_stage != "A":  # resume: weights saved at the end of the previous stage
        model.load_weights(f"{out_prefix}.stage{order[order.index(start_stage) - 1]}.weights.h5")
    for stage, steps, lr, hard in [("A", dc.steps_a, dc.lr_a, False), ("B", dc.steps_b, dc.lr_b, True)]:
        if steps <= 0 or order.index(stage) < order.index(start_stage):
            continue
        variables = model.trainable_variables + projector.trainable_variables
        opt = _optimizer(lr, steps, min(200, steps // 10), tc.weight_decay)
        opt.build(variables)

        @tf.function(input_signature=[PAIR_SIGNATURE, DISTILL_SIGNATURE], reduce_retracing=True)
        def step_ab(pairs, distill):
            with tf.GradientTape() as tape:
                lp = pair_loss(model, pairs)
                ld = distill_loss(model, projector, distill)
                total = lp + dc.distill_weight * ld
            grads = tape.gradient(total, variables)
            opt.apply_gradients([(g, v) for g, v in zip(grads, variables) if g is not None])
            return {"pair": lp, "distill": ld, "total": total}

        running, t0 = {}, time.time()
        for step in range(1, steps + 1):
            pairs = sample_pairs(rng, codec, cache, dc.n_pairs, hard)
            distill = sample_distill(rng, codec, cache, dc.n_distill_u, dc.n_distill_s)
            losses = step_ab({k: tf.constant(v) for k, v in pairs.items()}, {k: tf.constant(v) for k, v in distill.items()})
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v)
            if step % tc.log_every == 0:
                info["history"].append({"stage": stage, "step": step, **{k: v / tc.log_every for k, v in running.items()}})
                _log(stage, step, steps, running, tc.log_every, t0)
                running = {}
        model.save_weights(f"{out_prefix}.stage{stage}.weights.h5")
        info[f"stage_{stage}_seconds"] = time.time() - t0

    # ---- stage C: multi-task + distillation + contrastive replay
    variables = model.trainable_variables + projector.trainable_variables
    opt = _optimizer(tc.learning_rate, tc.steps, tc.warmup_steps, tc.weight_decay)
    opt.build(variables)
    task_utters = tc.n_tool + tc.n_extract + tc.n_classify + 2 * tc.n_embed
    n_replay = max(4, round(task_utters * dc.replay_frac / (1 - dc.replay_frac)))
    full = cfg.binding_cycles
    step_fns = {}

    def make_step(cycles: int):
        @tf.function(input_signature=[BATCH_SIGNATURE, PAIR_SIGNATURE, DISTILL_SIGNATURE], reduce_retracing=True)
        def step_c(batch, pairs, distill):
            with tf.GradientTape() as tape:
                losses, _ = compute_losses(model, batch, cycles, training=True)
                losses["replay"] = pair_loss(model, pairs)
                losses["distill"] = distill_loss(model, projector, distill)
                losses["total"] = (losses["total"] + dc.replay_weight * losses["replay"]
                                   + dc.distill_weight_c * losses["distill"])
            grads = tape.gradient(losses["total"], variables)
            opt.apply_gradients([(g, v) for g, v in zip(grads, variables) if g is not None])
            return losses
        return step_c

    running, t0 = {}, time.time()
    weights_path = out_prefix + ".weights.h5"
    for step in range(1, tc.steps + 1):
        cycles = full if full == 1 or rng.random() < tc.full_cycles_prob else rng.randint(1, full - 1)
        if cycles not in step_fns:
            step_fns[cycles] = make_step(cycles)
        batch = sample_batch(rng, codec, lib, tc.n_tool, tc.n_extract, tc.n_embed, n_classify=tc.n_classify)
        pairs = sample_pairs(rng, codec, cache, n_replay, hard=True)
        distill = sample_distill(rng, codec, cache, dc.n_distill_u // 2, dc.n_distill_s // 2)
        losses = step_fns[cycles](*({k: tf.constant(v) for k, v in d.items()} for d in (batch, pairs, distill)))
        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + float(v)
        if step % tc.log_every == 0:
            info["history"].append({"stage": "C", "step": step, **{k: v / tc.log_every for k, v in running.items()}})
            _log("C", step, tc.steps, running, tc.log_every, t0, f"cycles {cycles} ")
            running = {}
        if step % tc.save_every == 0 or step == tc.steps:
            model.save_weights(weights_path)
    info["stage_C_seconds"] = time.time() - t0
    info["train_seconds"] = time.time() - t_all
    return info


# ------------------------------------------------------------ real-data eval
def evaluate_real_intents(agent, clinc_path: str, n: int = 300, n_labels: int = 5, seed: int = 11,
                          cycles: Optional[int] = None) -> Dict[str, float]:
    """Zero-shot classification of real CLINC150 test utterances whose intents were held
    out of every training corpus: gold intent + random held-out distractors."""
    rng = random.Random(seed)
    intents = real_eval_intents(clinc_path)
    with open(clinc_path, encoding="utf-8") as f:
        test = [(t, lab) for t, lab in json.load(f)["test"] if lab in set(intents)]
    hits = []
    for text, lab in rng.sample(test, min(n, len(test))):
        labels = [lab] + rng.sample([i for i in intents if i != lab], n_labels - 1)
        rng.shuffle(labels)
        out = agent.classify(text, [_humanize(x) for x in labels], task="Label the intent of the request.", cycles=cycles)
        hits.append(out["label"] == _humanize(lab))
    return {"n": len(hits), "n_labels": n_labels, "n_intents": len(intents), "accuracy": float(np.mean(hits)),
            "chance": 1.0 / n_labels}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the ARC 1 distillation corpus (texts.json + pairs.json)")
    ap.add_argument("--out", default="data/arc1_teacher")
    ap.add_argument("--external-dir", default="data/external")
    ap.add_argument("--n-tool", type=int, default=20000)
    ap.add_argument("--n-classify", type=int, default=6000)
    a = ap.parse_args()
    print(json.dumps(build_corpus(a.out, a.external_dir, a.n_tool, a.n_classify), indent=2))
