#!/usr/bin/env python3
"""Same-checkpoint ARC 1 RSAA ablation with a conservative replacement gate.

Each candidate loads identical weights, receives its own held-out temperature
calibration, and is evaluated on the deterministic ARC 1 tool, extraction, and
classification sets.  The JSON output is suitable for release review; this
script never overwrites the deployed model artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gpbacay_arcane.arc1 import Arc1Config, Arc1Model
from gpbacay_arcane.arc1_codec import Arc1Codec
from gpbacay_arcane.arc1_data import build_tool_library
from gpbacay_arcane.arc1_train import calibrate, full_evaluation
from gpbacay_arcane.tokenization import BytePairTokenizer


QUALITY_PATHS = {
    "tool_selection": ("tools_heldout_values", "tool_selection_acc"),
    "exact_call": ("tools_heldout_values", "exact_call_acc"),
    "unseen_tool_selection": ("tools_unseen_tools", "tool_selection_acc"),
    "unseen_exact_call": ("tools_unseen_tools", "exact_call_acc"),
    "extraction_f1": ("extraction_heldout_values", "field_f1"),
    "classification": ("classify_heldout", "accuracy"),
    "unseen_classification": ("classify_unseen_tools", "accuracy"),
}


def _summary(evaluation: Dict[str, Any], binding_cycles: int) -> Dict[str, Any]:
    full = evaluation[f"cycles_{binding_cycles}"]
    quality = {name: float(full[section][metric]) for name, (section, metric) in QUALITY_PATHS.items()}
    latency = {
        "tools_p50_ms": float(full["tools_heldout_values"]["latency_ms_p50"]),
        "tools_p90_ms": float(full["tools_heldout_values"]["latency_ms_p90"]),
        "classify_p50_ms": float(full["classify_heldout"]["latency_ms_p50"]),
    }
    return {
        "quality": quality,
        "quality_macro": sum(quality.values()) / len(quality),
        "latency": latency,
        "latency_macro_ms": (latency["tools_p50_ms"] + latency["classify_p50_ms"]) / 2.0,
    }


def _gate(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    deltas = {
        key: candidate["quality"][key] - baseline["quality"][key]
        for key in baseline["quality"]
    }
    quality_gain = candidate["quality_macro"] - baseline["quality_macro"]
    latency_ratio = candidate["latency_macro_ms"] / max(baseline["latency_macro_ms"], 1e-9)
    passed = quality_gain >= 0.01 and min(deltas.values()) >= -0.01 and latency_ratio <= 2.0
    return {
        "passed": passed,
        "criteria": {
            "minimum_macro_quality_gain": 0.01,
            "maximum_single_metric_regression": 0.01,
            "maximum_latency_ratio": 2.0,
        },
        "quality_macro_delta": quality_gain,
        "metric_deltas": deltas,
        "latency_ratio": latency_ratio,
    }


def _candidate(base: Dict[str, Any], weights: str, tokenizer, rsaa_cycles: int,
               n_tool: int, n_extract: int) -> Dict[str, Any]:
    payload = dict(base)
    payload.update(
        rsaa_cycles=rsaa_cycles,
        active_rsaa_cycles=rsaa_cycles,
        rsaa_calibration={},
    )
    cfg = Arc1Config.from_dict(payload)
    model = Arc1Model(cfg).build_model()
    model.load_weights(weights)

    started = time.perf_counter()
    calibration = calibrate(
        model,
        Arc1Codec(tokenizer, cfg.seq_len, cfg.schema_len),
        build_tool_library(),
    )
    cfg.rsaa_calibration = {str(rsaa_cycles): dict(cfg.calibration)}
    evaluation = full_evaluation(
        model,
        tokenizer,
        n_tool=n_tool,
        n_extract=n_extract,
        cycles_list=[cfg.binding_cycles],
    )
    return {
        "rsaa_cycles": rsaa_cycles,
        "config": cfg.to_dict(),
        "calibration": calibration,
        "evaluation": evaluation,
        "summary": _summary(evaluation, cfg.binding_cycles),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _parse_cycles(values: Iterable[str]) -> list[int]:
    cycles = sorted({int(value) for value in values})
    if not cycles or cycles[0] != 0 or any(value < 0 for value in cycles):
        raise ValueError("--rsaa-cycles must include the zero-cycle baseline")
    return cycles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Models/arc1_arc1_tiny.config.json",
                        help="Candidate configuration")
    parser.add_argument("--weights", default="Models/arc1_arc1_tiny.weights.h5",
                        help="Candidate weights")
    parser.add_argument("--baseline-config", default=None,
                        help="Optional legacy config; candidate cycle 1..N is compared with its cycle 0")
    parser.add_argument("--baseline-weights", default=None,
                        help="Optional legacy weights (requires --baseline-config)")
    parser.add_argument("--tokenizer", default="Models/arc1_arc1_tiny_tokenizer.json")
    parser.add_argument("--out", default="Models/arc1_rsaa_comparison.json")
    parser.add_argument("--rsaa-cycles", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--n-tool", type=int, default=300)
    parser.add_argument("--n-extract", type=int, default=150)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        base = json.load(handle)
    baseline_base = base
    baseline_weights = args.weights
    if bool(args.baseline_config) != bool(args.baseline_weights):
        parser.error("--baseline-config and --baseline-weights must be supplied together")
    if args.baseline_config:
        with open(args.baseline_config, encoding="utf-8") as handle:
            baseline_base = json.load(handle)
        baseline_weights = args.baseline_weights
    tokenizer = BytePairTokenizer.load(args.tokenizer)
    cycles = _parse_cycles(args.rsaa_cycles)
    results: Dict[str, Any] = {}
    for value in cycles:
        print(f"[rsaa] evaluating cycles={value}", flush=True)
        candidate_config = baseline_base if value == 0 else base
        candidate_weights = baseline_weights if value == 0 else args.weights
        results[str(value)] = _candidate(
            candidate_config, candidate_weights, tokenizer, value, args.n_tool, args.n_extract
        )
        results[str(value)]["checkpoint"] = candidate_weights
        print(json.dumps(results[str(value)]["summary"], indent=2), flush=True)

    baseline = results["0"]["summary"]
    gates = {key: _gate(baseline, result["summary"]) for key, result in results.items() if key != "0"}
    passing = [int(key) for key, gate in gates.items() if gate["passed"]]
    winner = max(
        passing,
        key=lambda key: results[str(key)]["summary"]["quality_macro"],
        default=0,
    )
    report = {
        "baseline_checkpoint": baseline_weights,
        "candidate_checkpoint": args.weights,
        "evaluation_sizes": {"tools": args.n_tool, "extraction_and_classification": args.n_extract},
        "candidates": results,
        "replacement_gates": gates,
        "winner_rsaa_cycles": winner,
        "replace_legacy": winner > 0,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"winner_rsaa_cycles": winner, "replace_legacy": winner > 0, "gates": gates}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
