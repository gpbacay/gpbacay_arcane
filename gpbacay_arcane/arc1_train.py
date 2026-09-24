"""Multi-task training, calibration, and evaluation for ARC 1 decision heads.

Losses per step (one shared encoder pass over every sequence in the batch):

* noul    — BCE on tool relevance, optional-argument presence, boolean values
* span    — CE on start/end pointers over the COPY tokens
* choice  — CE over enum candidate groups
* embed   — InfoNCE between two paraphrases of the same intent (echo pooling)
* lm      — optional auxiliary next-token CE (``lm_weight``, off by default: on
            this data it stays near uniform and only costs compute)

A ladder depth is sampled per step so every nested slice is trained.
After training, one scalar temperature per head is fit on held-out data
(Laya-style calibration) and ECE before/after is reported.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from .arc1 import Arc1Config, Arc1Model
from .arc1_codec import SPAN_TYPES, Arc1Codec, EncodedSeq, pad_batch
from .arc1_data import (
    EXTRACT_TOOL_NAME,
    HELD_OUT_TOOLS,
    Example,
    ExtractExample,
    build_tool_library,
    extract_tool_spec,
    fixed_eval_set,
    fixed_extract_set,
    sample_extract_example,
    sample_tool_example,
    _single,
)
from .tools import Arc1Agent, ToolParam, ToolSpec


# ------------------------------------------------------------------ batching
@dataclass
class _Rows:
    seqs: List[EncodedSeq] = field(default_factory=list)
    noul_idx: List[int] = field(default_factory=list)
    noul_label: List[float] = field(default_factory=list)
    span_idx: List[int] = field(default_factory=list)
    span_start: List[int] = field(default_factory=list)
    span_end: List[int] = field(default_factory=list)
    choice_groups: List[List[int]] = field(default_factory=list)
    choice_label: List[int] = field(default_factory=list)
    embed_a: List[int] = field(default_factory=list)
    embed_b: List[int] = field(default_factory=list)
    embed_group: List[int] = field(default_factory=list)

    def add(self, seq: EncodedSeq) -> int:
        self.seqs.append(seq)
        return len(self.seqs) - 1


def _add_param_rows(rows: _Rows, codec: Arc1Codec, tool_name: str, param: ToolParam, text: str,
                    value: Any, span: Optional[Tuple[int, int]], present: bool) -> None:
    seq = codec.arg_seq(tool_name, param.name, param.type, param.description, param.required, text)
    idx = rows.add(seq)
    ptype = (param.type or "string").lower()
    if ptype in ("boolean", "bool"):
        rows.noul_idx.append(idx)
        rows.noul_label.append(1.0 if bool(value) else 0.0)
        return
    rows.noul_idx.append(idx)
    rows.noul_label.append(1.0 if present else 0.0)
    if not present:
        return
    if param.enum:
        group = [rows.add(codec.enum_seq(tool_name, param.name, param.description, str(o), text)) for o in param.enum]
        rows.choice_groups.append(group)
        rows.choice_label.append(list(param.enum).index(value))
    elif ptype in SPAN_TYPES and span is not None:
        tok = codec.char_span_to_tokens(seq, text, span)
        if tok is not None:
            rows.span_idx.append(idx)
            rows.span_start.append(tok[0])
            rows.span_end.append(tok[1])


def add_tool_example(rows: _Rows, codec: Arc1Codec, ex: Example) -> None:
    called = {c.tool: c for c in ex.calls}
    for spec in ex.tools:
        idx = rows.add(codec.tool_seq(spec.name, spec.description, ex.user))
        rows.noul_idx.append(idx)
        rows.noul_label.append(1.0 if spec.name in called else 0.0)
    for spec in ex.tools:
        call = called.get(spec.name)
        if call is None:
            continue
        for param in spec.parameters:
            present = param.name in call.arguments
            _add_param_rows(rows, codec, spec.name, param, ex.user, call.arguments.get(param.name),
                            call.spans.get(param.name), present)


def add_extract_example(rows: _Rows, codec: Arc1Codec, ex: ExtractExample) -> None:
    spec = extract_tool_spec(ex.schema)
    for param in spec.parameters:
        present = param.name in ex.record
        _add_param_rows(rows, codec, EXTRACT_TOOL_NAME, param, ex.text, ex.record.get(param.name),
                        ex.spans.get(param.name), present)


def add_embed_pairs(rows: _Rows, codec: Arc1Codec, rng: random.Random, lib, n_pairs: int, split: str = "train") -> None:
    names = [n for n in lib if n not in HELD_OUT_TOOLS]
    for gid, name in enumerate(rng.sample(names, min(n_pairs, len(names)))):
        a, _ = _single(rng, lib[name], split)
        b, _ = _single(rng, lib[name], split)
        rows.embed_a.append(rows.add(codec.embed_seq(a)))
        rows.embed_b.append(rows.add(codec.embed_seq(b)))
        rows.embed_group.append(gid)


def rows_to_tensors(rows: _Rows, seq_len: int) -> Dict[str, np.ndarray]:
    # Two length buckets (short half / long half) so short rows are not padded
    # to the longest compound utterance; ``order`` restores original row order.
    lengths = np.asarray([len(s.ids) for s in rows.seqs])
    by_len = np.argsort(lengths, kind="stable")
    half = max(1, len(by_len) // 2)
    short, long_ = by_len[:half], by_len[half:]
    if len(long_) == 0:
        short, long_ = by_len[:-1], by_len[-1:]
    ids_a = pad_batch([rows.seqs[i].ids for i in short], multiple=32, max_len=seq_len)
    ids_b = pad_batch([rows.seqs[i].ids for i in long_], multiple=32, max_len=seq_len)
    order = np.argsort(np.concatenate([short, long_]), kind="stable").astype(np.int32)
    width = max(ids_a.shape[1], ids_b.shape[1])
    span_mask = np.zeros((len(rows.span_idx), width), dtype=bool)
    for r, idx in enumerate(rows.span_idx):
        seq = rows.seqs[idx]
        span_mask[r, seq.span_lo : min(seq.span_hi, width)] = True
    kmax = max((len(g) for g in rows.choice_groups), default=1)
    groups = np.full((len(rows.choice_groups), kmax), -1, dtype=np.int32)
    for r, g in enumerate(rows.choice_groups):
        groups[r, : len(g)] = g
    embed_rows = rows.embed_a + rows.embed_b
    pool_mask = np.zeros((len(embed_rows), width), dtype=bool)
    for r, idx in enumerate(embed_rows):
        seq = rows.seqs[idx]
        pool_mask[r, seq.span_lo : min(seq.span_hi, width)] = True
    return {
        "ids_a": ids_a,
        "ids_b": ids_b,
        "order": order,
        "noul_idx": np.asarray(rows.noul_idx, dtype=np.int32),
        "noul_label": np.asarray(rows.noul_label, dtype=np.float32),
        "span_idx": np.asarray(rows.span_idx, dtype=np.int32),
        "span_start": np.asarray(rows.span_start, dtype=np.int32),
        "span_end": np.asarray(rows.span_end, dtype=np.int32),
        "span_mask": span_mask,
        "choice_groups": groups,
        "choice_label": np.asarray(rows.choice_label, dtype=np.int32),
        "embed_rows": np.asarray(embed_rows, dtype=np.int32),
        "embed_pool": pool_mask,
        "embed_group": np.asarray(rows.embed_group, dtype=np.int32),
    }


def sample_batch(rng: random.Random, codec: Arc1Codec, lib, n_tool: int, n_extract: int, n_embed: int,
                 split: str = "train") -> Dict[str, np.ndarray]:
    rows = _Rows()
    for _ in range(n_tool):
        add_tool_example(rows, codec, sample_tool_example(rng, lib, split))
    for _ in range(n_extract):
        add_extract_example(rows, codec, sample_extract_example(rng, split))
    if n_embed:
        add_embed_pairs(rows, codec, rng, lib, n_embed, split)
    return rows_to_tensors(rows, codec.seq_len)


# -------------------------------------------------------------------- losses
_NEG = -1e9

BATCH_SIGNATURE = {
    "ids_a": tf.TensorSpec([None, None], tf.int32),
    "ids_b": tf.TensorSpec([None, None], tf.int32),
    "order": tf.TensorSpec([None], tf.int32),
    "noul_idx": tf.TensorSpec([None], tf.int32),
    "noul_label": tf.TensorSpec([None], tf.float32),
    "span_idx": tf.TensorSpec([None], tf.int32),
    "span_start": tf.TensorSpec([None], tf.int32),
    "span_end": tf.TensorSpec([None], tf.int32),
    "span_mask": tf.TensorSpec([None, None], tf.bool),
    "choice_groups": tf.TensorSpec([None, None], tf.int32),
    "choice_label": tf.TensorSpec([None], tf.int32),
    "embed_rows": tf.TensorSpec([None], tf.int32),
    "embed_pool": tf.TensorSpec([None, None], tf.bool),
    "embed_group": tf.TensorSpec([None], tf.int32),
}


def _safe_mean(x):
    return tf.math.divide_no_nan(tf.reduce_sum(x), tf.cast(tf.size(x), x.dtype))


def _lm_sums(model: Arc1Model, ids, lm_logits):
    target = ids[:, 1:]
    mask = tf.cast(tf.not_equal(target, model.arc1_config.pad_id), tf.float32)
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=lm_logits[:, :-1])
    return tf.reduce_sum(ce * mask), tf.reduce_sum(mask)


def decide_buckets(model: Arc1Model, batch, depth: int, training: bool, with_lm: bool):
    """Run both length buckets, re-pad to a shared width, restore row order."""
    outs = [
        model.decide(batch[key], training=training, depth=depth, with_lm=with_lm)
        for key in ("ids_a", "ids_b")
    ]
    width = tf.maximum(tf.shape(batch["ids_a"])[1], tf.shape(batch["ids_b"])[1])
    merged = {}
    for key in ("noul", "choice"):
        merged[key] = tf.gather(tf.concat([o[key] for o in outs], axis=0), batch["order"])
    for key in ("span_start", "span_end", "hidden"):
        parts = []
        for o in outs:
            t = o[key]
            pad = width - tf.shape(t)[1]
            paddings = [[0, 0], [0, pad]] + [[0, 0]] * (len(t.shape) - 2)
            parts.append(tf.pad(t, paddings))
        merged[key] = tf.gather(tf.concat(parts, axis=0), batch["order"])
    if with_lm:
        sums = [_lm_sums(model, batch[k], o["lm"]) for k, o in zip(("ids_a", "ids_b"), outs)]
        merged["lm_loss"] = tf.math.divide_no_nan(sums[0][0] + sums[1][0], sums[0][1] + sums[1][1])
    return merged


def compute_losses(model: Arc1Model, batch, depth: int, training: bool, lm_weight: float = 0.0,
                   embed_temperature: float = 0.05):
    out = decide_buckets(model, batch, depth, training, with_lm=lm_weight > 0)
    losses = {}

    noul = tf.gather(out["noul"], batch["noul_idx"])
    losses["noul"] = _safe_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch["noul_label"], logits=noul))

    def span_ce(logits, target):
        rows = tf.gather(logits, batch["span_idx"])
        rows = tf.where(batch["span_mask"], rows, _NEG)
        return _safe_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=rows))

    losses["span"] = 0.5 * (span_ce(out["span_start"], batch["span_start"]) + span_ce(out["span_end"], batch["span_end"]))

    groups = batch["choice_groups"]
    valid = groups >= 0
    cand = tf.gather(out["choice"], tf.maximum(groups, 0))
    cand = tf.where(valid, cand, _NEG)
    losses["choice"] = _safe_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch["choice_label"], logits=cand)
    )

    n_pairs = tf.shape(batch["embed_group"])[0]
    hidden = tf.gather(out["hidden"], batch["embed_rows"])
    emb = model.pooled_embedding(hidden, batch["embed_pool"])
    a, b = emb[:n_pairs], emb[n_pairs:]
    sim = tf.matmul(a, b, transpose_b=True) / embed_temperature
    same = tf.equal(batch["embed_group"][:, None], batch["embed_group"][None, :])
    eye = tf.eye(n_pairs, dtype=tf.bool)
    sim = tf.where(tf.logical_and(same, tf.logical_not(eye)), _NEG, sim)
    labels = tf.range(n_pairs)
    losses["embed"] = 0.5 * (
        _safe_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=sim))
        + _safe_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=tf.transpose(sim)))
    )

    losses["lm"] = out["lm_loss"] if lm_weight > 0 else tf.constant(0.0)

    total = losses["noul"] + losses["span"] + losses["choice"] + 0.5 * losses["embed"] + lm_weight * losses["lm"]
    losses["total"] = total
    return losses, out


# --------------------------------------------------------------- calibration
def _nll_sigmoid(logits, labels, t):
    z = logits / t
    return float(np.mean(np.logaddexp(0.0, z) - labels * z))


def _nll_softmax(logit_rows, targets, t):
    total = 0.0
    for row, y in zip(logit_rows, targets):
        z = row / t
        m = z.max()
        total += (m + math.log(np.exp(z - m).sum())) - z[y]
    return total / max(len(targets), 1)


MIN_CALIBRATION_SAMPLES = 50


def _fit_temperature(nll_fn, n: int) -> float:
    """Grid-search the NLL-optimal temperature; keep 1.0 when data is too thin."""
    if n < MIN_CALIBRATION_SAMPLES:
        return 1.0
    grid = np.exp(np.linspace(np.log(0.2), np.log(8.0), 80))
    scores = [nll_fn(t) for t in grid]
    return float(grid[int(np.argmin(scores))])


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """ECE over the predicted-class confidence of binary decisions."""
    conf = np.where(probs >= 0.5, probs, 1 - probs)
    correct = ((probs >= 0.5).astype(np.float32) == labels).astype(np.float32)
    edges = np.linspace(0.5, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(ece)


def _enum_batch(rng: random.Random, codec: Arc1Codec, lib, n: int) -> Dict[str, np.ndarray]:
    """Examples drawn only from tools with enum parameters (rare in the normal mix)."""
    enum_tools = [t for t in lib.values() if t.fixed and any(p.enum for p in t.params) and t.name not in HELD_OUT_TOOLS]
    rows = _Rows()
    for _ in range(n):
        tool = rng.choice(enum_tools)
        while True:
            text, call = _single(rng, tool, "eval")
            if any(p.enum and p.name in call.arguments for p in tool.params):
                break
        add_tool_example(rows, codec, Example(text, [tool.spec()], [call]))
    return rows_to_tensors(rows, codec.seq_len)


def calibrate(model: Arc1Model, codec: Arc1Codec, lib, n_batches: int = 12, seed: int = 99) -> Dict[str, Any]:
    """Fit one temperature per head on held-out values (split='eval')."""
    rng = random.Random(seed)
    noul_l, noul_y, span_rows, span_y, choice_rows, choice_y = [], [], [], [], [], []
    batches = [sample_batch(rng, codec, lib, n_tool=10, n_extract=4, n_embed=0, split="eval") for _ in range(n_batches)]
    batches += [_enum_batch(rng, codec, lib, 16) for _ in range(4)]
    for batch in batches:
        out = decide_buckets(model, {k: tf.constant(v) for k, v in batch.items()}, None, False, with_lm=False)
        noul = out["noul"].numpy()
        noul_l.extend(noul[batch["noul_idx"]].tolist())
        noul_y.extend(batch["noul_label"].tolist())
        for key, tgt in (("span_start", "span_start"), ("span_end", "span_end")):
            logits = out[key].numpy()
            for r, idx in enumerate(batch["span_idx"]):
                row = logits[idx][batch["span_mask"][r]]
                lo = int(np.argmax(batch["span_mask"][r]))
                span_rows.append(row)
                span_y.append(int(batch[tgt][r]) - lo)
        choice = out["choice"].numpy()
        for g, y in zip(batch["choice_groups"], batch["choice_label"]):
            g = g[g >= 0]
            choice_rows.append(choice[g])
            choice_y.append(int(y))
    noul_l, noul_y = np.asarray(noul_l), np.asarray(noul_y)
    temps = {
        "noul": _fit_temperature(lambda t: _nll_sigmoid(noul_l, noul_y, t), len(noul_y)),
        "span": _fit_temperature(lambda t: _nll_softmax(span_rows, span_y, t), len(span_y) // 2),
        "choice": _fit_temperature(lambda t: _nll_softmax(choice_rows, choice_y, t), len(choice_y)),
    }
    raw_p = 1 / (1 + np.exp(-noul_l))
    cal_p = 1 / (1 + np.exp(-noul_l / temps["noul"]))
    report = {
        "temperatures": temps,
        "noul_ece_before": expected_calibration_error(raw_p, noul_y),
        "noul_ece_after": expected_calibration_error(cal_p, noul_y),
        "n_noul": int(len(noul_y)),
        "n_span": int(len(span_y) // 2),
        "n_choice": int(len(choice_y)),
    }
    model.arc1_config.calibration = dict(temps)
    return report


# ---------------------------------------------------------------- evaluation
def _norm(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 4)
    return str(v).strip().lower()


def evaluate_tools(agent: Arc1Agent, examples: Sequence[Example], depth: Optional[int] = None) -> Dict[str, float]:
    n = len(examples)
    tool_ok = call_ok = none_n = none_ok = arg_total = arg_ok = 0
    latencies = []
    for ex in examples:
        out = agent.run(ex.user, tools=ex.tools, execute=False, depth=depth)
        latencies.append(out["latency_ms"])
        pred = {c["name"]: c["arguments"] for c in out["function_calls"]}
        gold = {c.tool: c.arguments for c in ex.calls}
        tool_ok += set(pred) == set(gold)
        exact = set(pred) == set(gold)
        for name, args in gold.items():
            for k, v in args.items():
                arg_total += 1
                got = pred.get(name, {}).get(k, None)
                ok = got is not None and _norm(got) == _norm(v)
                arg_ok += ok
                exact = exact and ok
            if name in pred and set(pred[name]) != set(args):
                exact = False
        call_ok += exact
        if not gold:
            none_n += 1
            none_ok += not pred
    return {
        "n": n,
        "tool_selection_acc": tool_ok / max(n, 1),
        "exact_call_acc": call_ok / max(n, 1),
        "argument_acc": arg_ok / max(arg_total, 1),
        "no_tool_acc": none_ok / max(none_n, 1),
        "latency_ms_p50": float(np.median(latencies)) if latencies else 0.0,
    }


def evaluate_extract(agent: Arc1Agent, examples: Sequence[ExtractExample], depth: Optional[int] = None) -> Dict[str, float]:
    tp = fp = fn = exact = 0
    for ex in examples:
        rec = agent.extract(ex.text, ex.schema, depth=depth)["record"]
        gold = {k: _norm(v) for k, v in ex.record.items()}
        pred = {k: _norm(v) for k, v in rec.items()}
        for k, v in pred.items():
            if gold.get(k) == v:
                tp += 1
            else:
                fp += 1
        fn += sum(1 for k, v in gold.items() if pred.get(k) != v)
        exact += pred == gold
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "n": len(examples),
        "field_precision": precision,
        "field_recall": recall,
        "field_f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "exact_record_acc": exact / max(len(examples), 1),
    }


def evaluate_embeddings(agent: Arc1Agent, lib, n: int = 60, seed: int = 7) -> Dict[str, float]:
    """Retrieval@1: does a paraphrase's nearest neighbour share its intent?"""
    rng = random.Random(seed)
    names = [k for k in lib if k not in HELD_OUT_TOOLS]
    texts, labels = [], []
    for i in range(n):
        name = names[i % len(names)]
        texts.append(_single(rng, lib[name], "eval")[0])
        labels.append(name)
    vecs = np.asarray([agent.embed(t) for t in texts])
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -np.inf)
    nn = sims.argmax(axis=1)
    return {"n": n, "intent_retrieval_at_1": float(np.mean([labels[i] == labels[j] for i, j in enumerate(nn)]))}


def full_evaluation(model: Arc1Model, tokenizer, n_tool: int = 300, n_extract: int = 150,
                    depths: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    agent = Arc1Agent(model, tokenizer)
    lib = build_tool_library()
    eval_set = fixed_eval_set("eval", n_tool)
    unseen = fixed_eval_set("unseen_tools", max(n_tool // 2, 50))
    extract_set = fixed_extract_set(n_extract)
    report: Dict[str, Any] = {}
    for depth in depths or [model.arc1_config.num_layers]:
        report[f"depth_{depth}"] = {
            "tools_heldout_values": evaluate_tools(agent, eval_set, depth),
            "tools_unseen_tools": evaluate_tools(agent, unseen, depth),
            "extraction_heldout_values": evaluate_extract(agent, extract_set, depth),
        }
    report["embeddings"] = evaluate_embeddings(agent, lib)
    return report


# ------------------------------------------------------------------ training
@dataclass
class TrainConfig:
    steps: int = 3000
    learning_rate: float = 1e-3
    warmup_steps: int = 150
    weight_decay: float = 0.01
    n_tool: int = 12
    n_extract: int = 4
    n_embed: int = 8
    lm_weight: float = 0.0
    full_depth_prob: float = 0.6
    log_every: int = 25
    save_every: int = 250
    seed: int = 0


def train(model: Arc1Model, tokenizer, tc: TrainConfig, out_prefix: str,
          resume: bool = False) -> Dict[str, Any]:
    cfg = model.arc1_config
    codec = Arc1Codec(tokenizer, cfg.seq_len)
    lib = build_tool_library()
    rng = random.Random(tc.seed)
    tf.random.set_seed(tc.seed)

    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=tc.learning_rate * 0.05,
        decay_steps=max(tc.steps - tc.warmup_steps, 1),
        alpha=0.05,
        warmup_target=tc.learning_rate,
        warmup_steps=tc.warmup_steps,
    )
    opt = tf.keras.optimizers.AdamW(learning_rate=schedule, weight_decay=tc.weight_decay, clipnorm=1.0)
    # Build on every variable: shallow ladder steps only produce gradients for a subset.
    opt.build(model.trainable_variables)

    depths = sorted(set(int(d) for d in cfg.ladder_depths if 1 <= int(d) <= cfg.num_layers) | {cfg.num_layers})
    step_fns = {}

    def make_step(depth: int):
        @tf.function(input_signature=[BATCH_SIGNATURE])
        def step(batch):
            with tf.GradientTape() as tape:
                losses, _ = compute_losses(model, batch, depth, training=True, lm_weight=tc.lm_weight)
            variables = model.trainable_variables
            grads = tape.gradient(losses["total"], variables)
            pairs = [(g, v) for g, v in zip(grads, variables) if g is not None]
            opt.apply_gradients(pairs)
            return losses
        return step

    weights_path = out_prefix + ".weights.h5"
    if resume and os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"[arc1] resumed weights from {weights_path}", flush=True)

    history, running, t0 = [], {}, time.time()
    for step in range(1, tc.steps + 1):
        depth = cfg.num_layers if rng.random() < tc.full_depth_prob or len(depths) == 1 else rng.choice(depths[:-1])
        if depth not in step_fns:
            step_fns[depth] = make_step(depth)
        batch = sample_batch(rng, codec, lib, tc.n_tool, tc.n_extract, tc.n_embed)
        losses = step_fns[depth]({k: tf.constant(v) for k, v in batch.items()})
        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + float(v)
        if step % tc.log_every == 0:
            avg = {k: v / tc.log_every for k, v in running.items()}
            running = {}
            elapsed = time.time() - t0
            eta = elapsed / step * (tc.steps - step)
            history.append({"step": step, **avg})
            print(
                f"[arc1] step {step}/{tc.steps} depth {depth} "
                + " ".join(f"{k}={v:.3f}" for k, v in avg.items())
                + f" | {elapsed / step:.2f}s/step eta {eta / 60:.1f}m",
                flush=True,
            )
        if step % tc.save_every == 0 or step == tc.steps:
            model.save_weights(weights_path)
    return {"history": history, "train_seconds": time.time() - t0}


def save_artifacts(model: Arc1Model, tokenizer, out_prefix: str, metrics: Dict[str, Any]) -> Dict[str, str]:
    paths = {
        "weights": out_prefix + ".weights.h5",
        "config": out_prefix + ".config.json",
        "tokenizer": out_prefix + "_tokenizer.json",
        "metrics": out_prefix + ".metrics.json",
    }
    model.save_weights(paths["weights"])
    with open(paths["config"], "w", encoding="utf-8") as f:
        json.dump(model.arc1_config.to_dict(), f, indent=2)
    tokenizer.save(paths["tokenizer"])
    with open(paths["metrics"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return paths
