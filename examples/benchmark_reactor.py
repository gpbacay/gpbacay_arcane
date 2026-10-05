#!/usr/bin/env python3
"""Reactor benchmark: ARC 1 alone vs memory alone vs ARC 1 + memory (same weights, no fine-tuning).

1. Real text: CLINC150 test utterances for the intents held out of every ARC 1 training set, 5 labels
   (the protocol of ``evaluate_real_intents``). The memory holds ``shots`` CLINC *train* utterances per intent.
2. Unseen tools: the held-out tool split. The memory holds ``shots`` example requests per held-out tool,
   rendered from the training templates and values (the eval uses held-out ones).
3. Scale (memory only, no model): all 150 CLINC150 intents, up to 100 train utterances each (15,000
   examples), 1,000 test utterances: insert and lookup cost, 150-way accuracy.

  curl -L -o data/external/clinc150_data_full.json https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json
  python examples/benchmark_reactor.py --shots 1 2 5 10
"""
import argparse, json, os, random, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

from gpbacay_arcane import Reactor, load_arc1
from gpbacay_arcane.arc1_data import HELD_OUT_TOOLS, _single, build_tool_library, fixed_eval_set
from gpbacay_arcane.arc1_distill import _humanize, real_eval_intents
from gpbacay_arcane.arc1_train import evaluate_tools

CLINC = os.path.join(ROOT, "data", "external", "clinc150_data_full.json")
TASK = "Label the intent of the request."


def clinc(agent, shots, n, seed=11):
    data = json.load(open(CLINC, encoding="utf-8"))
    intents = real_eval_intents(CLINC)
    both, memory, rng = Reactor(agent), Reactor(), random.Random(0)
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
    labels_only, with_args = Reactor(agent), Reactor(agent)
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
    by, rng, reactor = {}, random.Random(0), Reactor()
    for t, lab in data["train"]:
        by.setdefault(lab, []).append(t)
    pairs = [(t, lab) for lab, ts in by.items() for t in rng.sample(ts, min(shots, len(ts)))]
    t0 = time.perf_counter()
    for t, lab in pairs:
        reactor.remember(t, lab)
    insert_ms = (time.perf_counter() - t0) * 1000 / len(pairs)
    labels, hits, lat = sorted(by), [], []
    for t, lab in random.Random(1).sample(data["test"], n):
        t0 = time.perf_counter()
        hits.append(reactor.classify(t, labels)["label"] == lab)
        lat.append((time.perf_counter() - t0) * 1000)
    return {"examples": len(pairs), "labels": len(labels), "accuracy": round(float(np.mean(hits)), 3),
            "insert_ms": round(insert_ms, 3), "classify_ms_p50": round(float(np.median(lat)), 2),
            "classify_ms_p90": round(float(np.percentile(lat, 90)), 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, nargs="+", default=[5])
    ap.add_argument("-n", type=int, default=600)
    a = ap.parse_args()
    if os.path.exists(CLINC):
        print("scale, memory only, CLINC150 150-way:", json.dumps(scale()), flush=True)
    agent = load_arc1()
    for shots in a.shots:
        if os.path.exists(CLINC):
            print(f"CLINC150 unseen intents, 5-way, {shots}-shot memory:", json.dumps(clinc(agent, shots, a.n)), flush=True)
        print(f"unseen tools, {shots}-shot memory:", json.dumps(unseen_tools(agent, shots, min(a.n, 300))), flush=True)
