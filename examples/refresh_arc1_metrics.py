#!/usr/bin/env python3
"""Re-evaluate the shipped ARC 1 weights with the current decoder and rewrite only the metrics' ``evaluation``.

Use after a decoding change (tools.py / arc1_codec.py) when the weights did not change. Unlike
``train_arc1.py --eval-only`` it does not re-fit calibration or rewrite the config, so the weights'
fingerprint and every published copy (.rcn, ONNX) stay valid. Run on an idle machine: latency is measured.

    python examples/refresh_arc1_metrics.py
    cd arcane-docs-web && npm run sync:arc1-metrics
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # serving setting, so latency matches the API

from gpbacay_arcane.arc1 import Arc1Config, Arc1Model  # noqa: E402
from gpbacay_arcane.arc1_distill import evaluate_real_intents  # noqa: E402
from gpbacay_arcane.arc1_train import full_evaluation  # noqa: E402
from gpbacay_arcane.tokenization import BytePairTokenizer  # noqa: E402
from gpbacay_arcane.tools import Arc1Agent  # noqa: E402

PREFIX = os.path.join(ROOT, "Models", "arc1_arc1_tiny")
CLINC = os.path.join(ROOT, "data", "external", "clinc150_data_full.json")


def main():
    with open(PREFIX + ".config.json", encoding="utf-8") as f:
        config = Arc1Config.from_dict(json.load(f))
    model = Arc1Model(config).build_model()
    model.load_weights(PREFIX + ".weights.h5")
    tokenizer = BytePairTokenizer.load(PREFIX + "_tokenizer.json")
    cycles = sorted({1, config.binding_cycles})
    evaluation = full_evaluation(model, tokenizer, n_tool=300, n_extract=150, cycles_list=cycles)
    if os.path.exists(CLINC):
        agent = Arc1Agent(model, tokenizer)
        evaluation["real_intents"] = {f"cycles_{c}": evaluate_real_intents(agent, CLINC, n=600, cycles=c) for c in cycles}
    with open(PREFIX + ".metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    metrics["evaluation"] = evaluation
    with open(PREFIX + ".metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()
