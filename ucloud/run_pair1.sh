#!/usr/bin/env bash
# Experiment 1 with Pair 1's code (robustness run), see pair1_kit/run_pair1_fix.sh.
# Needs Pair 1's code in $DATA_DIR/exp1:
#     mkdir -p $DATA_DIR/exp1 && cp -r $CODE_DIR/exp1_pair1_code/. $DATA_DIR/exp1/
# Optional: their own finished runs in $DATA_DIR/exp1/runs_official (then a padding check decides
# whether to keep them); without them the whole grid is run with the fix ("fresh", as for the results).
# Batch job: Extra options "RUN=pair1 PARALLEL=6".
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
