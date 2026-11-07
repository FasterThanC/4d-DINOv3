#!/usr/bin/env bash
# Experiment 1 (Pair 1's code) in our UCloud data folder: padding check + finish (see
# pair1_kit/run_pair1_fix.sh). Needs Pair 1's code in $DATA_DIR/exp1 and their finished runs in
# $DATA_DIR/exp1/runs_official. Batch job: Extra options "RUN=pair1".
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
source "$D/config.sh"
export EXP1="${EXP1:-$DATA_DIR/exp1}" FEAT="${FEAT:-$TEACHER_FEATS}"
[ -f "$EXP1/train.py" ] || { echo "Pair 1's code not found in $EXP1"; exit 1; }
B="$EXP1/src/backbones"
touch "$EXP1/src/__init__.py" "$B/__init__.py"     # make their `src` a normal package (no clash with ours)
[ -f "$B/ltae_official_backup.py" ] || cp "$B/ltae.py" "$B/ltae_official_backup.py"
cp "$CODE_DIR/pair1_kit/ltae.py" "$B/ltae.py"
cp "$CODE_DIR/pair1_kit/padding_check.py" "$EXP1/"
export WORKERS_PER_RUN="${WORKERS_PER_RUN:-$(( WORKERS / ${PARALLEL:-6} > 0 ? WORKERS / ${PARALLEL:-6} : 1 ))}"
bash "$CODE_DIR/pair1_kit/run_pair1_fix.sh"
