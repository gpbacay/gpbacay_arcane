#!/bin/sh
# Controlled A/B: same warm start, seed, steps, lr; only --probe-attention differs.
# Sequential on purpose: two runs side by side exhaust 8 GB of RAM and swap (~10x slower).
set -e
cd "$(dirname "$0")/../.."
INIT=Models/arc1_arc1_tiny.weights.h5
COMMON="--init-weights $INIT --steps 1000 --learning-rate 5e-4 --seed 0 --skip-eval --out-dir Models/ab_probe_attention"
python -u examples/train_arc1.py $COMMON --run-name control > Models/ab_probe_attention/control.log 2>&1
python -u examples/train_arc1.py $COMMON --run-name probe_attention --probe-attention > Models/ab_probe_attention/probe_attention.log 2>&1
echo done
