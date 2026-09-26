"""Multi-task training, calibration, and evaluation for ARC 1 (Resonant Schema Binding).

One step perceives every utterance and every unique schema text in the batch
once, composes all probes, binds them, and sums four losses:

* fire    — BCE: tool fires, optional argument present, boolean value
* anchor  — CE on start/end pointers over utterance tokens
* select  — CE over each enum argument's (or classification label set's) option probes
* embed   — InfoNCE between two paraphrases of the same intent

A binding cycle count is sampled per step so every ``cycles`` setting works
with the same weights. After training, one temperature per readout is fitted
on held-out values and ECE before/after is reported.
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

from .arc1 import Arc1Model
from .arc1_codec import (
    ROLE_BOOL,
    ROLE_OPTION,
    ROLE_TOOL,
    SPAN_TYPES,
    Arc1Codec,
    Utterance,
    option_text,
    pad_batch,
    param_text,
    role_for,
    tool_text,
)
from .arc1_data import (
    ClassifyExample,
    EXTRACT_TOOL_DESCRIPTION,
    EXTRACT_TOOL_NAME,
    HELD_OUT_TOOLS,
    Example,
    ExtractExample,
    build_tool_library,
    classify_tool_spec,
    extract_tool_spec,
    fixed_classify_set,
    fixed_eval_set,
    fixed_extract_set,
    sample_classify_example,
    sample_extract_example,
    sample_tool_example,
    _single,
)
from .tools import Arc1Agent, ToolParam


# ------------------------------------------------------------------ batching
@dataclass
class _Rows:
    utters: List[Utterance] = field(default_factory=list)
    schema_index: Dict[str, int] = field(default_factory=dict)
    probe_example: List[int] = field(default_factory=list)
    probe_a: List[int] = field(default_factory=list)
    probe_b: List[int] = field(default_factory=list)
    probe_role: List[int] = field(default_factory=list)
    fire_idx: List[int] = field(default_factory=list)
    fire_label: List[float] = field(default_factory=list)
    anchor_idx: List[int] = field(default_factory=list)
    anchor_start: List[int] = field(default_factory=list)
    anchor_end: List[int] = field(default_factory=list)
    select_param: List[int] = field(default_factory=list)
    select_options: List[List[int]] = field(default_factory=list)
    select_label: List[int] = field(default_factory=list)
    embed_a: List[int] = field(default_factory=list)
    embed_b: List[int] = field(default_factory=list)
    embed_group: List[int] = field(default_factory=list)

    def utter(self, utt: Utterance) -> int:
        self.utters.append(utt)
        return len(self.utters) - 1

    def _schema(self, text: str) -> int:
        return self.schema_index.setdefault(text, len(self.schema_index))

    def probe(self, example: int, text_a: str, text_b: Optional[str], role: int) -> int:
        self.probe_example.append(example)
        self.probe_a.append(self._schema(text_a))
        self.probe_b.append(-1 if text_b is None else self._schema(text_b))
        self.probe_role.append(role)
        return len(self.probe_role) - 1

    def fire(self, idx: int, label: float) -> None:
        self.fire_idx.append(idx)
        self.fire_label.append(float(label))


def _add_param_rows(rows: _Rows, codec: Arc1Codec, u: int, utt: Utterance, ttext: str, param: ToolParam,
                    text: str, value: Any, span: Optional[Tuple[int, int]], present: bool) -> None:
    ptext = param_text(param.name, param.type, param.description, param.required)
    role = role_for(param.type, param.enum)
    idx = rows.probe(u, ptext, ttext, role)
    if role == ROLE_BOOL:
        rows.fire(idx, 1.0 if bool(value) else 0.0)
        return
    if not param.required:
        rows.fire(idx, 1.0 if present else 0.0)
    if not present:
        return
    if param.enum:
        hints = param.enum_descriptions or {}
        group = [rows.probe(u, option_text(o, hints.get(str(o))), ptext, ROLE_OPTION) for o in param.enum]
        rows.select_param.append(idx)
        rows.select_options.append(group)
        rows.select_label.append(list(param.enum).index(value))
    elif (param.type or "string").lower() in SPAN_TYPES and span is not None:
        tok = codec.char_span_to_tokens(utt, text, span)
        if tok is not None:
            rows.anchor_idx.append(idx)
            rows.anchor_start.append(tok[0])
            rows.anchor_end.append(tok[1])


def add_tool_example(rows: _Rows, codec: Arc1Codec, ex: Example) -> None:
    utt = codec.utterance(ex.user)
    u = rows.utter(utt)
    called = {c.tool: c for c in ex.calls}
    for spec in ex.tools:
        rows.fire(rows.probe(u, tool_text(spec.name, spec.description), None, ROLE_TOOL),
                  1.0 if spec.name in called else 0.0)
    for spec in ex.tools:
        call = called.get(spec.name)
        if call is None:
            continue
        ttext = tool_text(spec.name, spec.description)
        for param in spec.parameters:
            _add_param_rows(rows, codec, u, utt, ttext, param, ex.user, call.arguments.get(param.name),
                            call.spans.get(param.name), param.name in call.arguments)


def add_extract_example(rows: _Rows, codec: Arc1Codec, ex: ExtractExample) -> None:
    utt = codec.utterance(ex.text)
    u = rows.utter(utt)
    spec = extract_tool_spec(ex.schema)
    ttext = tool_text(EXTRACT_TOOL_NAME, EXTRACT_TOOL_DESCRIPTION)
    for param in spec.parameters:
        _add_param_rows(rows, codec, u, utt, ttext, param, ex.text, ex.record.get(param.name),
                        ex.spans.get(param.name), param.name in ex.record)


def add_classify_example(rows: _Rows, codec: Arc1Codec, ex: ClassifyExample) -> None:
    utt = codec.utterance(ex.text)
    u = rows.utter(utt)
    spec = classify_tool_spec(ex.labels, ex.task, ex.descriptions)
    _add_param_rows(rows, codec, u, utt, tool_text(spec.name, spec.description), spec.parameters[0],
                    ex.text, ex.label, None, True)


def add_embed_pairs(rows: _Rows, codec: Arc1Codec, rng: random.Random, lib, n_pairs: int, split: str = "train") -> None:
    names = [n for n in lib if n not in HELD_OUT_TOOLS]
    for gid, name in enumerate(rng.sample(names, min(n_pairs, len(names)))):
        rows.embed_a.append(rows.utter(codec.utterance(_single(rng, lib[name], split)[0])))
        rows.embed_b.append(rows.utter(codec.utterance(_single(rng, lib[name], split)[0])))
        rows.embed_group.append(gid)


def rows_to_tensors(rows: _Rows, codec: Arc1Codec) -> Dict[str, np.ndarray]:
    schema_texts = sorted(rows.schema_index, key=rows.schema_index.get)
    kmax = max((len(g) for g in rows.select_options), default=1)
    options = np.full((len(rows.select_options), kmax), -1, dtype=np.int32)
    for r, g in enumerate(rows.select_options):
        options[r, : len(g)] = g
    i32 = lambda xs: np.asarray(xs, dtype=np.int32)  # noqa: E731
    return {
        "utter_ids": pad_batch([u.ids for u in rows.utters], multiple=16, max_len=codec.seq_len),
        "schema_ids": pad_batch([codec.schema(t) for t in schema_texts], multiple=16, max_len=codec.schema_len),
        "probe_example": i32(rows.probe_example),
        "probe_a": i32(rows.probe_a),
        "probe_b": i32(rows.probe_b),
        "probe_role": i32(rows.probe_role),
        "fire_idx": i32(rows.fire_idx),
        "fire_label": np.asarray(rows.fire_label, dtype=np.float32),
        "anchor_idx": i32(rows.anchor_idx),
        "anchor_start": i32(rows.anchor_start),
        "anchor_end": i32(rows.anchor_end),
        "select_param": i32(rows.select_param),
        "select_options": options,
        "select_label": i32(rows.select_label),
        "embed_a": i32(rows.embed_a),
        "embed_b": i32(rows.embed_b),
        "embed_group": i32(rows.embed_group),
    }


def sample_batch(rng: random.Random, codec: Arc1Codec, lib, n_tool: int, n_extract: int, n_embed: int,
                 split: str = "train", n_classify: int = 0) -> Dict[str, np.ndarray]:
    rows = _Rows()
    for _ in range(n_tool):
        add_tool_example(rows, codec, sample_tool_example(rng, lib, split))
    for _ in range(n_extract):
        add_extract_example(rows, codec, sample_extract_example(rng, split))
    for _ in range(n_classify):
        add_classify_example(rows, codec, sample_classify_example(rng, lib, split))
    if n_embed:
        add_embed_pairs(rows, codec, rng, lib, n_embed, split)
    return rows_to_tensors(rows, codec)


# -------------------------------------------------------------------- losses
_NEG = -1e9

BATCH_SIGNATURE = {
    "utter_ids": tf.TensorSpec([None, None], tf.int32),
    "schema_ids": tf.TensorSpec([None, None], tf.int32),
    "probe_example": tf.TensorSpec([None], tf.int32),
    "probe_a": tf.TensorSpec([None], tf.int32),
    "probe_b": tf.TensorSpec([None], tf.int32),
    "probe_role": tf.TensorSpec([None], tf.int32),
    "fire_idx": tf.TensorSpec([None], tf.int32),
    "fire_label": tf.TensorSpec([None], tf.float32),
    "anchor_idx": tf.TensorSpec([None], tf.int32),
    "anchor_start": tf.TensorSpec([None], tf.int32),
    "anchor_end": tf.TensorSpec([None], tf.int32),
    "select_param": tf.TensorSpec([None], tf.int32),
    "select_options": tf.TensorSpec([None, None], tf.int32),
    "select_label": tf.TensorSpec([None], tf.int32),
    "embed_a": tf.TensorSpec([None], tf.int32),
    "embed_b": tf.TensorSpec([None], tf.int32),
    "embed_group": tf.TensorSpec([None], tf.int32),
}


def _safe_mean(x):
    return tf.math.divide_no_nan(tf.reduce_sum(x), tf.cast(tf.size(x), x.dtype))


def forward_batch(model: Arc1Model, batch, cycles: Optional[int], training: bool):
    """Perceive schemas and utterances once, bind every probe; add enum logits."""
    engrams = model.schema_engrams(batch["schema_ids"], training=training)
    engram_a = tf.gather(engrams, batch["probe_a"])
    has_b = batch["probe_b"] >= 0
    engram_b = tf.where(has_b[:, None], tf.gather(engrams, tf.maximum(batch["probe_b"], 0)), 0.0)
    out = model.decide(batch["utter_ids"], batch["probe_example"], engram_a, engram_b, batch["probe_role"],
                       cycles=cycles, training=training, with_embedding=True)
    options = batch["select_options"]
    q = tf.gather(out["select_q"], batch["select_param"])[:, None, :]
    k = tf.gather(out["select_k"], tf.maximum(options, 0))
    out["select"] = tf.where(options >= 0, model.select_logits(q, k), _NEG)
    return out


def compute_losses(model: Arc1Model, batch, cycles: Optional[int], training: bool, embed_temperature: float = 0.05):
    out = forward_batch(model, batch, cycles, training)
    losses = {}
    fire = tf.gather(out["fire"], batch["fire_idx"])
    losses["fire"] = _safe_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch["fire_label"], logits=fire))

    def anchor_ce(logits, target):
        rows = tf.gather(logits, batch["anchor_idx"])
        return _safe_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=rows))

    losses["anchor"] = 0.5 * (anchor_ce(out["anchor_start"], batch["anchor_start"])
                              + anchor_ce(out["anchor_end"], batch["anchor_end"]))
    losses["select"] = _safe_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch["select_label"], logits=out["select"])
    )

    emb = out["embedding"]
    a = tf.gather(emb, batch["embed_a"])
    b = tf.gather(emb, batch["embed_b"])
    n_pairs = tf.shape(batch["embed_group"])[0]
    sim = tf.matmul(a, b, transpose_b=True) / embed_temperature
    same = tf.equal(batch["embed_group"][:, None], batch["embed_group"][None, :])
    sim = tf.where(tf.logical_and(same, tf.logical_not(tf.eye(n_pairs, dtype=tf.bool))), _NEG, sim)
    labels = tf.range(n_pairs)
    losses["embed"] = 0.5 * (
        _safe_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=sim))
        + _safe_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=tf.transpose(sim)))
    )
    losses["total"] = losses["fire"] + losses["anchor"] + losses["select"] + 0.5 * losses["embed"]
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
    return rows_to_tensors(rows, codec)


def calibrate(model: Arc1Model, codec: Arc1Codec, lib, n_batches: int = 12, seed: int = 99) -> Dict[str, Any]:
    """Fit one temperature per readout on held-out values (split='eval')."""
    rng = random.Random(seed)
    fire_l, fire_y, anchor_rows, anchor_y, select_rows, select_y = [], [], [], [], [], []
    batches = [sample_batch(rng, codec, lib, n_tool=12, n_extract=6, n_embed=0, split="eval", n_classify=6)
               for _ in range(n_batches)]
    batches += [_enum_batch(rng, codec, lib, 16) for _ in range(4)]
    for batch in batches:
        out = forward_batch(model, {k: tf.constant(v) for k, v in batch.items()}, None, False)
        fire_l.extend(out["fire"].numpy()[batch["fire_idx"]].tolist())
        fire_y.extend(batch["fire_label"].tolist())
        for key in ("anchor_start", "anchor_end"):
            logits = out[key].numpy()
            for r, idx in enumerate(batch["anchor_idx"]):
                row = logits[idx]
                valid = row > _NEG / 2
                anchor_rows.append(row[valid])
                anchor_y.append(int(batch[key][r]) - int(np.argmax(valid)))
        select = out["select"].numpy()
        for row, g, y in zip(select, batch["select_options"], batch["select_label"]):
            select_rows.append(row[g >= 0])
            select_y.append(int(y))
    fire_l, fire_y = np.asarray(fire_l), np.asarray(fire_y)
    temps = {
        "fire": _fit_temperature(lambda t: _nll_sigmoid(fire_l, fire_y, t), len(fire_y)),
        "anchor": _fit_temperature(lambda t: _nll_softmax(anchor_rows, anchor_y, t), len(anchor_y) // 2),
        "select": _fit_temperature(lambda t: _nll_softmax(select_rows, select_y, t), len(select_y)),
    }
    raw_p = 1 / (1 + np.exp(-fire_l))
    cal_p = 1 / (1 + np.exp(-fire_l / temps["fire"]))
    report = {
        "temperatures": temps,
        "fire_ece_before": expected_calibration_error(raw_p, fire_y),
        "fire_ece_after": expected_calibration_error(cal_p, fire_y),
        "n_fire": int(len(fire_y)),
        "n_anchor": int(len(anchor_y) // 2),
        "n_select": int(len(select_y)),
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


def evaluate_tools(agent: Arc1Agent, examples: Sequence[Example], cycles: Optional[int] = None) -> Dict[str, float]:
    n = len(examples)
    tool_ok = call_ok = none_n = none_ok = arg_total = arg_ok = 0
    latencies = []
    for ex in examples:
        out = agent.run(ex.user, tools=ex.tools, execute=False, cycles=cycles)
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
    warm = latencies[5:] or latencies  # first requests include tracing + schema encoding
    return {
        "n": n,
        "tool_selection_acc": tool_ok / max(n, 1),
        "exact_call_acc": call_ok / max(n, 1),
        "argument_acc": arg_ok / max(arg_total, 1),
        "no_tool_acc": none_ok / max(none_n, 1),
        "latency_ms_p50": float(np.median(warm)) if warm else 0.0,
        "latency_ms_p90": float(np.percentile(warm, 90)) if warm else 0.0,
    }


def evaluate_extract(agent: Arc1Agent, examples: Sequence[ExtractExample], cycles: Optional[int] = None) -> Dict[str, float]:
    tp = fp = fn = exact = 0
    for ex in examples:
        rec = agent.extract(ex.text, ex.schema, cycles=cycles)["record"]
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


def evaluate_classify(agent: Arc1Agent, examples: Sequence[ClassifyExample], cycles: Optional[int] = None) -> Dict[str, Any]:
    hits: Dict[str, List[bool]] = {}
    latencies = []
    for ex in examples:
        out = agent.classify(ex.text, ex.labels, task=ex.task, cycles=cycles, descriptions=ex.descriptions)
        latencies.append(out["latency_ms"])
        hits.setdefault(ex.kind, []).append(_norm(out["label"]) == _norm(ex.label))
    every = [h for v in hits.values() for h in v]
    warm = latencies[5:] or latencies
    return {
        "n": len(examples),
        "accuracy": float(np.mean(every)) if every else 0.0,
        **{f"accuracy_{k}": float(np.mean(v)) for k, v in sorted(hits.items())},
        "latency_ms_p50": float(np.median(warm)) if warm else 0.0,
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
                    cycles_list: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    agent = Arc1Agent(model, tokenizer)
    lib = build_tool_library()
    eval_set = fixed_eval_set("eval", n_tool)
    unseen = fixed_eval_set("unseen_tools", max(n_tool // 2, 50))
    extract_set = fixed_extract_set(n_extract)
    classify_set = fixed_classify_set("eval", n_extract)
    classify_unseen = fixed_classify_set("unseen_tools", max(n_extract // 2, 50))
    report: Dict[str, Any] = {}
    for cycles in cycles_list or [model.arc1_config.binding_cycles]:
        report[f"cycles_{cycles}"] = {
            "tools_heldout_values": evaluate_tools(agent, eval_set, cycles),
            "tools_unseen_tools": evaluate_tools(agent, unseen, cycles),
            "extraction_heldout_values": evaluate_extract(agent, extract_set, cycles),
            "classify_heldout": evaluate_classify(agent, classify_set, cycles),
            "classify_unseen_tools": evaluate_classify(agent, classify_unseen, cycles),
        }
    report["embeddings"] = evaluate_embeddings(agent, lib)
    return report


# ------------------------------------------------------------------ training
@dataclass
class TrainConfig:
    steps: int = 4000
    learning_rate: float = 2e-3
    warmup_steps: int = 200
    weight_decay: float = 0.01
    n_tool: int = 16
    n_extract: int = 6
    n_embed: int = 8
    n_classify: int = 6
    full_cycles_prob: float = 0.6
    log_every: int = 50
    save_every: int = 500
    seed: int = 0


def train(model: Arc1Model, tokenizer, tc: TrainConfig, out_prefix: str, resume: bool = False) -> Dict[str, Any]:
    cfg = model.arc1_config
    codec = Arc1Codec(tokenizer, cfg.seq_len, cfg.schema_len)
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
    opt.build(model.trainable_variables)

    full = cfg.binding_cycles
    step_fns = {}

    def make_step(cycles: int):
        @tf.function(input_signature=[BATCH_SIGNATURE], reduce_retracing=True)
        def step(batch):
            with tf.GradientTape() as tape:
                losses, _ = compute_losses(model, batch, cycles, training=True)
            variables = model.trainable_variables
            grads = tape.gradient(losses["total"], variables)
            opt.apply_gradients([(g, v) for g, v in zip(grads, variables) if g is not None])
            return losses
        return step

    weights_path = out_prefix + ".weights.h5"
    if resume and os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"[arc1] resumed weights from {weights_path}", flush=True)

    history, running, t0 = [], {}, time.time()
    for step in range(1, tc.steps + 1):
        cycles = full if full == 1 or rng.random() < tc.full_cycles_prob else rng.randint(1, full - 1)
        if cycles not in step_fns:
            step_fns[cycles] = make_step(cycles)
        batch = sample_batch(rng, codec, lib, tc.n_tool, tc.n_extract, tc.n_embed, n_classify=tc.n_classify)
        losses = step_fns[cycles]({k: tf.constant(v) for k, v in batch.items()})
        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + float(v)
        if step % tc.log_every == 0:
            avg = {k: v / tc.log_every for k, v in running.items()}
            running = {}
            elapsed = time.time() - t0
            eta = elapsed / step * (tc.steps - step)
            history.append({"step": step, **avg})
            print(
                f"[arc1] step {step}/{tc.steps} cycles {cycles} "
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
