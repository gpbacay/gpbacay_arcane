#!/bin/sh
# Controlled A/B: same warm start, seed, steps, lr; only --lateral-inhibition differs.
set -e
cd "$(dirname "$0")/../.."
INIT=Models/distill3/arc1_arc1_tiny.weights.h5
COMMON="--init-weights $INIT --steps 1000 --learning-rate 5e-4 --seed 0 --out-dir Models/ab_inhibition"
python -u examples/train_arc1.py $COMMON --run-name control > Models/ab_inhibition/control.log 2>&1
python -u examples/train_arc1.py $COMMON --run-name inhibition --lateral-inhibition > Models/ab_inhibition/inhibition.log 2>&1
echo done
