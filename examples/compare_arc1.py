#!/usr/bin/env python3
"""Evaluate several ARC 1 checkpoints under identical conditions and print a table.

Each checkpoint is a prefix with ``.weights.h5``, ``.config.json`` (with fitted
calibration), and ``_tokenizer.json``. Runs the standard evaluation, a larger
one for tighter estimates, and zero-shot classification on real CLINC150
utterances whose intents no checkpoint trained on.

  python examples/compare_arc1.py old=Models/arc1_arc1_tiny new=Models/distill/arc1_arc1_tiny
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # serving-like single-request latency

from gpbacay_arcane.arc1 import Arc1Config, Arc1Model
from gpbacay_arcane.arc1_distill import evaluate_real_intents
from gpbacay_arcane.arc1_train import full_evaluation
from gpbacay_arcane.tokenization import BytePairTokenizer
from gpbacay_arcane.tools import Arc1Agent

CLINC = "data/external/clinc150_data_full.json"
ROWS = [
    ("tool selection", "tools_heldout_values", "tool_selection_acc"),
    ("exact call", "tools_heldout_values", "exact_call_acc"),
    ("no-tool refusal", "tools_heldout_values", "no_tool_acc"),
    ("unseen-tool selection", "tools_unseen_tools", "tool_selection_acc"),
    ("unseen-tool exact call", "tools_unseen_tools", "exact_call_acc"),
    ("extraction F1", "extraction_heldout_values", "field_f1"),
    ("extraction exact record", "extraction_heldout_values", "exact_record_acc"),
    ("classification (held-out)", "classify_heldout", "accuracy"),
    ("  support routing", "classify_heldout", "accuracy_support"),
    ("  product topics", "classify_heldout", "accuracy_topic_products"),
    ("  sentiment", "classify_heldout", "accuracy_sentiment"),
    ("classification (unseen tools)", "classify_unseen_tools", "accuracy"),
    ("latency p50 ms", "tools_heldout_values", "latency_ms_p50"),
]


def load(prefix: str):
    with open(prefix + ".config.json", encoding="utf-8") as f:
        model = Arc1Model(Arc1Config.from_dict(json.load(f)))
    model.build_model()
    model.load_weights(prefix + ".weights.h5")
    return model, BytePairTokenizer.load(prefix + "_tokenizer.json")


def main():
    named = [a.split("=", 1) for a in sys.argv[1:]]
    results = {}
    for name, prefix in named:
        model, tok = load(prefix)
        cycles = model.arc1_config.binding_cycles
        print(f"[compare] {name}: {prefix} ({model.count_params():,} params)", flush=True)
        results[name] = {
            "standard": full_evaluation(model, tok, n_tool=300, n_extract=150, cycles_list=sorted({1, cycles})),
            "large": full_evaluation(model, tok, n_tool=1000, n_extract=500, cycles_list=[1]),
            "real_intents": {f"cycles_{c}": evaluate_real_intents(Arc1Agent(model, tok), CLINC, n=600, cycles=c)
                             for c in sorted({1, cycles})},
        }
    os.makedirs("Models/compare", exist_ok=True)
    with open("Models/compare/arc1_compare.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    for split in ("standard", "large"):
        print(f"\n## {split} evaluation (cycles 1)")
        print("| metric | " + " | ".join(n for n, _ in named) + " |")
        print("|---|" + "---|" * len(named))
        for label, sect, key in ROWS:
            vals = []
            for n, _ in named:
                v = results[n][split]["cycles_1"][sect].get(key)
                vals.append("-" if v is None else (f"{v:.1f}" if "latency" in key else f"{100 * v:.1f}%"))
            print(f"| {label} | " + " | ".join(vals) + " |")
    print("\n## real CLINC150 utterances, unseen intents (5 labels, chance 20%)")
    for n, _ in named:
        print(n, {c: f"{100 * r['accuracy']:.1f}% (n={r['n']})" for c, r in results[n]["real_intents"].items()})
    print("\nwrote Models/compare/arc1_compare.json")


if __name__ == "__main__":
    main()
