#!/usr/bin/env python3
"""Hippocampus benchmark: ARC 1 alone vs memory alone vs ARC 1 + memory (same weights, no fine-tuning).

1. Real text: CLINC150 test utterances for the intents held out of every ARC 1 training set, 5 labels
   (the protocol of ``evaluate_real_intents``). The memory holds ``shots`` CLINC *train* utterances per intent.
2. Unseen tools: the held-out tool split. The memory holds ``shots`` example requests per held-out tool,
   rendered from the training templates and values (the eval uses held-out ones).
3. Scale (memory only, no model): all 150 CLINC150 intents, up to 100 train utterances each (15,000
   examples), 1,000 test utterances: insert and lookup cost, 150-way accuracy.

  curl -L -o data/external/clinc150_data_full.json https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json
  python examples/benchmark_hippocampus.py --shots 1 2 5 10
"""
import argparse, json, os, random, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

from gpbacay_arcane import Hippocampus, load_arc1
from gpbacay_arcane.arc1_data import HELD_OUT_TOOLS, _single, build_tool_library, fixed_eval_set
from gpbacay_arcane.arc1_distill import _humanize, real_eval_intents
from gpbacay_arcane.arc1_train import evaluate_tools

CLINC = os.path.join(ROOT, "data", "external", "clinc150_data_full.json")
TASK = "Label the intent of the request."


def clinc(agent, shots, n, seed=11):
    data = json.load(open(CLINC, encoding="utf-8"))
    intents = real_eval_intents(CLINC)
    both, memory, rng = Hippocampus(agent), Hippocampus(), random.Random(0)
    for i in intents:
        for text in rng.sample([t for t, lab in data["train"] if lab == i], shots):
            both.remember(text, _humanize(i))
            memory.remember(text, _humanize(i))
    rng = random.Random(seed)
    test = [(t, lab) for t, lab in data["test"] if lab in set(intents)]
    hits = {"arc1": [], "memory": [], "arc1+memory": []}
    for text, lab in rng.sample(test, min(n, len(test))):
        labels = [lab] + rng.sample([i for i in intents if i != lab], 4)
        rng.shuffle(labels)
        labels, gold = [_humanize(x) for x in labels], _humanize(lab)
        hits["arc1"].append(agent.classify(text, labels, task=TASK)["label"] == gold)
        hits["memory"].append(memory.classify(text, labels)["label"] == gold)
        hits["arc1+memory"].append(both.classify(text, labels, task=TASK)["label"] == gold)
    return {"n": len(hits["arc1"]), **{k: round(float(np.mean(v)), 3) for k, v in hits.items()}}


def unseen_tools(agent, shots, n):
    lib, rng = build_tool_library(), random.Random(0)
    labels_only, with_args = Hippocampus(agent), Hippocampus(agent)
    for name in HELD_OUT_TOOLS:
        for _ in range(shots):
            text, call = _single(rng, lib[name], "train")
            labels_only.remember(text, name)
            with_args.remember(text, name, call.arguments)
    examples = fixed_eval_set("unseen_tools", n)
    keys = ("tool_selection_acc", "exact_call_acc", "argument_acc", "no_tool_acc")
    pick = lambda m: {k: round(m[k], 3) for k in keys}
    return {"n": n, "arc1": pick(evaluate_tools(agent, examples)),
            "arc1+memory": pick(evaluate_tools(labels_only, examples)),
            "arc1+memory+arguments": pick(evaluate_tools(with_args, examples))}


def scale(shots=100, n=1000):
    data = json.load(open(CLINC, encoding="utf-8"))
    by, rng, hippocampus = {}, random.Random(0), Hippocampus()
    for t, lab in data["train"]:
        by.setdefault(lab, []).append(t)
    pairs = [(t, lab) for lab, ts in by.items() for t in rng.sample(ts, min(shots, len(ts)))]
    t0 = time.perf_counter()
    for t, lab in pairs:
        hippocampus.remember(t, lab)
    insert_ms = (time.perf_counter() - t0) * 1000 / len(pairs)
    labels, hits, lat = sorted(by), [], []
    for t, lab in random.Random(1).sample(data["test"], n):
        t0 = time.perf_counter()
        hits.append(hippocampus.classify(t, labels)["label"] == lab)
        lat.append((time.perf_counter() - t0) * 1000)
    return {"examples": len(pairs), "labels": len(labels), "accuracy": round(float(np.mean(hits)), 3),
            "insert_ms": round(insert_ms, 3), "classify_ms_p50": round(float(np.median(lat)), 2),
            "classify_ms_p90": round(float(np.percentile(lat, 90)), 2)}


def policy(shots=20, n=1000, noise=0.2, repeats=3, seeds=(0, 1, 2)):
    """Memory only, CLINC150 150-way, mean over ``seeds``: ``remember`` alone vs ``observe``.

    Clean: each train utterance once, shuffled, with its true label. ``remember`` has no policy, so under a
    memory budget it can only keep a random subset; ``observe`` decides itself and caps at the same size.
    Noisy: every utterance is seen ``repeats`` times in a shuffled stream and each sighting's label is
    independently wrong with probability ``noise`` (``remember`` keeps whichever sighting came last).
    """
    data = json.load(open(CLINC, encoding="utf-8"))
    by = {}
    for t, lab in data["train"]:
        by.setdefault(lab, []).append(t)
    labels, rows = sorted(by), {}

    def run(name, memory, feed, source, test):
        for t, lab in source:
            feed(memory, t, lab)
        acc = float(np.mean([memory.classify(t, labels)["label"] == lab for t, lab in test]))
        rows.setdefault(name, []).append((len(memory.labels()), acc))

    remember = lambda m, t, lab: m.remember(t, lab)
    observe = lambda m, t, lab: m.observe(t, lab)
    for seed in seeds:
        rng = random.Random(seed)
        pairs = [(t, lab) for lab, ts in by.items() for t in rng.sample(ts, min(shots, len(ts)))]
        rng.shuffle(pairs)
        test, budget = random.Random(seed + 100).sample(data["test"], n), len(pairs) // 2
        wrong = lambda lab: rng.choice([x for x in labels if x != lab]) if rng.random() < noise else lab
        stream = [(t, wrong(lab)) for t, lab in pairs for _ in range(repeats)]
        rng.shuffle(stream)
        run("clean: remember all", Hippocampus(), remember, pairs, test)
        run("clean: observe", Hippocampus(), observe, pairs, test)
        run(f"clean, budget {budget}: remember a random subset", Hippocampus(), remember, rng.sample(pairs, budget), test)
        run(f"clean, budget {budget}: observe, max_size", Hippocampus(max_size=budget), observe, pairs, test)
        run(f"noisy {noise}: remember all", Hippocampus(), remember, stream, test)
        run(f"noisy {noise}: observe trust=.5", Hippocampus(threshold=0.9),
            lambda m, t, lab: m.observe(t, lab, trust=0.5), stream, test)
    return {k: {"examples": round(float(np.mean([r[0] for r in v]))), "accuracy": round(float(np.mean([r[1] for r in v])), 3),
                "per_seed": [round(r[1], 3) for r in v]} for k, v in rows.items()}


def teacher(agent, shots=(1, 5), per_intent=60, n=600, seeds=(0, 1, 2), accuracies=(1.0, 0.8)):
    """ARC 1 + memory on every CLINC150 held-out intent: ``shots`` hand-labeled examples each, then a stream of
    unlabeled utterances a *simulated* teacher may label (right with probability ``accuracy``, else a random
    other label). No real LLM was run: this measures the mechanism for a teacher of that quality.

    ``confident=0.8`` is the default: the teacher is asked when the decision is under it and the memory alone is
    not decisive. ``confident=1.0`` asks far more often. Accuracy is System 1 alone afterwards (teacher off) on
    test utterances, 29-way.
    """
    data = json.load(open(CLINC, encoding="utf-8"))
    intents = real_eval_intents(CLINC)
    labels = [_humanize(i) for i in intents]
    out = {}
    for k in shots:
        rows = {}
        for seed in seeds:
            rng = random.Random(seed)
            seeds_, stream, gold = [], [], {}
            for i in intents:
                pool = rng.sample([t for t, lab in data["train"] if lab == i], len([1 for t, lab in data["train"] if lab == i]))
                seeds_ += [(t, _humanize(i)) for t in pool[:k]]
                stream += [(t, _humanize(i)) for t in pool[k:k + per_intent]]
            gold = dict(stream)
            rng.shuffle(stream)
            test = rng.sample([(t, lab) for t, lab in data["test"] if lab in set(intents)], n)

            def build(teacher_fn=None, confident=0.8):
                h = Hippocampus(agent, teacher=teacher_fn, confident=confident)
                for t, lab in seeds_:
                    h.remember(t, lab)
                return h

            def noisy(accuracy):
                calls = {"n": 0}

                def ask(text, options):
                    calls["n"] += 1
                    right = gold[text]
                    return right if rng.random() < accuracy else rng.choice([x for x in options if x != right])
                return ask, calls

            def score(name, h, calls=None):
                h.teacher = None  # System 1 alone from here on
                acc = float(np.mean([h.classify(t, labels, task=TASK)["label"] == _humanize(lab) for t, lab in test]))
                taught = [d for d in h.labels() if h.graph.docs[d].get("source") == "auto"]
                right = sum(h.labels()[d] == gold.get(h.graph.nodes[d]["content"]) for d in taught)
                rows.setdefault(name, []).append((acc, calls["n"] if calls else 0, len(taught), right / max(len(taught), 1)))

            score("no teacher", build())
            for accuracy in accuracies:
                for name, confident in (("confident=1.0", 1.0), ("confident=0.8", 0.8)):
                    ask, calls = noisy(accuracy)
                    h = build(ask, confident)
                    for t, _ in stream:
                        h.classify(t, labels, task=TASK)
                    score(f"teacher {accuracy:.0%} right, {name}", h, calls)
            h = build()
            for t, lab in stream:
                h.remember(t, lab)
            score("every stream item labeled by hand", h)
        out[f"{k} hand-labeled per intent"] = {
            name: {"accuracy": round(float(np.mean([r[0] for r in v])), 3), "teacher_calls": round(float(np.mean([r[1] for r in v]))),
                   "taught": round(float(np.mean([r[2] for r in v]))), "taught_label_precision": round(float(np.mean([r[3] for r in v])), 3)}
            for name, v in rows.items()}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, nargs="+", default=[5])
    ap.add_argument("-n", type=int, default=600)
    ap.add_argument("--policy", action="store_true", help="only compare remember-all with observe (no ARC 1 needed)")
    ap.add_argument("--teacher", action="store_true", help="only measure learning from a simulated teacher (needs ARC 1)")
    a = ap.parse_args()
    if a.policy:
        for k, v in policy().items():
            print(f"{k}: {json.dumps(v)}", flush=True)
        sys.exit()
    if a.teacher:
        for k, rows in teacher(load_arc1()).items():
            print(k, flush=True)
            for name, v in rows.items():
                print(f"  {name}: {json.dumps(v)}", flush=True)
        sys.exit()
    if os.path.exists(CLINC):
        print("scale, memory only, CLINC150 150-way:", json.dumps(scale()), flush=True)
    agent = load_arc1()
    for shots in a.shots:
        if os.path.exists(CLINC):
            print(f"CLINC150 unseen intents, 5-way, {shots}-shot memory:", json.dumps(clinc(agent, shots, a.n)), flush=True)
        print(f"unseen tools, {shots}-shot memory:", json.dumps(unseen_tools(agent, shots, min(a.n, 300))), flush=True)
