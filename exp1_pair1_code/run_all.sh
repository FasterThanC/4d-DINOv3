#!/bin/bash
# Whole Experiment 1 grid on one GPU job. Finished runs are skipped.
set -euo pipefail
cd /work/ATDL_A3/exp1
export PASTIS=/work/ATDL_A3/data/PASTIS
export FEATS=/work/ATDL_A3/data/feats
export RUNS=/work/ATDL_A3/exp1/runs

python run_grid.py --pastis "$PASTIS" --feat_dir "$FEATS" --out "$RUNS" \
    --extra '--lr 3e-3 --epochs 80'

python run_grid.py --pastis "$PASTIS" --feat_dir "$FEATS" --out "$RUNS" \
    --single_best --extra '--lr 3e-3 --epochs 80'
