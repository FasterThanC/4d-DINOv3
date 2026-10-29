#!/usr/bin/env bash
# Whole Pair 2 pipeline after setup, download and a passed preflight. Each step is restartable,
# and after each step a check runs; the pipeline stops at the first failed check so no GPU time
# is spent on later steps built on a broken one.
#   bash ucloud/run_all.sh                       (inside tmux, or as a UCloud batch job:
#                                                 see ucloud/batch_full_run.sh)
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
source "$D/config.sh"
if [ ! -f "$DATA_DIR/preflight/PASSED" ] && [ "${FORCE:-0}" != 1 ]; then
  echo "Run bash ucloud/preflight.sh first (or FORCE=1 to skip this guard)."; exit 1
fi
stage () { echo; echo "##### $1  ($(date '+%a %H:%M'))"; }

stage "1/4 DINOv3 features for all images"
bash "$D/02_extract_teacher.sh" full
python scripts/check_features.py --pastis "$PASTIS" --feats "$TEACHER_FEATS" --splits "$SPLITS"

stage "2/4 students 1 and 2 (train + save their tokens)"
bash "$D/03_train_students.sh" both ${STUDENT_ARGS:-}    # e.g. STUDENT_ARGS="--epochs 30"
python scripts/check_runs.py students --s1 "$RUNS/students/student1_copy" --s2 "$RUNS/students/student2_rebuild"
for s in student1_copy student2_rebuild; do
  python scripts/check_features.py --pastis "$PASTIS" --feats "$FEATS/$s" --splits "$SPLITS"
done

stage "3/4 L-TAE runs"
bash "$D/04_temporal_runs.sh"
n_done=$(ls "$RUNS"/temporal/*/results.json 2>/dev/null | wc -l)
echo "finished L-TAE runs: $n_done"

stage "4/4 NDVI probe, benchmark, comparisons"
bash "$D/05_probe_and_benchmark.sh"
echo "ALL DONE  ($(date '+%a %H:%M'))"
