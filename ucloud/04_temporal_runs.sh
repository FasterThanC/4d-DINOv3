#!/usr/bin/env bash
# L-TAE + decoder on each feature set: real vs shuffled dates, 100% vs 10% labels, 3 seeds.
# Several small runs share one GPU (PARALLEL). Finished runs are skipped, so it is restartable.
#   bash ucloud/04_temporal_runs.sh
#   FEATSETS="student1_copy student2_rebuild" SEEDS="0" bash ucloud/04_temporal_runs.sh
# DINOv3 runs are also made by Pair 1; keep them here only if both pairs use this same code.
set -euo pipefail
source "$(dirname "$0")/config.sh"
FEATSETS="${FEATSETS:-dinov3_sat student1_copy student2_rebuild}"
DATES="${DATES:-real shuffled}"
LABELS="${LABELS:-train train_10}"
SEEDS="${SEEDS:-0 1 2}"
PARALLEL="${PARALLEL:-3}"
TW=$(( WORKERS / PARALLEL > 6 ? 6 : (WORKERS / PARALLEL > 0 ? WORKERS / PARALLEL : 1) ))   # loader workers per run
STEPS="${STEPS:-5000}"
TOUT="${TEMPORAL_OUT:-$RUNS/temporal}"          # output folder of the runs
mkdir -p "$TOUT"

jobs_file="$TOUT/jobs.txt"
: > "$jobs_file"
for f in $FEATSETS; do for d in $DATES; do for l in $LABELS; do for s in $SEEDS; do
  echo "python scripts/train_temporal.py --pastis $PASTIS --feats $FEATS/$f --splits $SPLITS \
--out $TOUT --dates $d --train-split $l --seed $s --steps $STEPS --workers $TW \
> $TOUT/log_${f}_${d}_${l}_s${s}.txt 2>&1" >> "$jobs_file"
done; done; done; done
echo "$(wc -l < "$jobs_file") runs, $PARALLEL at a time"
xargs -P "$PARALLEL" -I{} bash -c "{}" < "$jobs_file" || \
  echo "WARNING: some runs failed; see $TOUT/log_*.txt. Rerun this script to retry them."
python scripts/summarize.py --runs "$TOUT" --out "$TOUT/summary.csv"
[ "$TOUT" = "$RUNS/temporal" ] && cp "$TOUT/summary.csv" "$RUNS/summary_temporal.csv" || true
