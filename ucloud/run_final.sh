#!/usr/bin/env bash
# Second, final round (after run_all.sh). Adds what makes the result solid:
#   1. NDVI probe again (bug fixed in scripts/ndvi_probe.py)
#   2. students 1 and 2 with two more training seeds (1, 2) + student 3 (rebuild only, control)
#   3. L-TAE with 3x longer training (15,000 steps) on all 8 feature sets, 3 seeds each,
#      into $RUNS/temporal_final (the first round in $RUNS/temporal is kept)
#   4. NDVI probe incl. student 3, and all comparisons -> $RUNS/logs/comparisons_final.log
# Restartable like run_all.sh. Batch job: Extra options "RUN=final PARALLEL=8".
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
source "$D/config.sh"
stage () { echo; echo "##### $1  ($(date '+%a %H:%M'))"; }
FINAL="$RUNS/temporal_final"

stage "1/4 NDVI probe (fixed)"
python scripts/ndvi_probe.py --pastis "$PASTIS" --splits "$SPLITS" --out "$RUNS/probe" \
  --feats dinov3="$TEACHER_FEATS" student1="$FEATS/student1_copy" student2="$FEATS/student2_rebuild" \
  2>&1 | tee "$RUNS/logs/probe.log"

stage "2/4 more students: seeds 1 and 2 of students 1 and 2, and student 3"
bash "$D/03_train_students.sh" seeds ${STUDENT_ARGS:-}
bash "$D/03_train_students.sh" student3 ${STUDENT_ARGS:-}
for sd in 1 2; do
  python scripts/check_runs.py students --s1 "$RUNS/students/student1_copy_s$sd" \
    --s2 "$RUNS/students/student2_rebuild_s$sd" --factor "${STUDENT_FACTOR:-0.9}"
done
for f in student1_copy_s1 student1_copy_s2 student2_rebuild_s1 student2_rebuild_s2 student3_rebuild_only; do
  python scripts/check_features.py --pastis "$PASTIS" --feats "$FEATS/$f" --splits "$SPLITS"
done

stage "3/4 L-TAE, 15,000 steps, 8 feature sets x real/shuffled x 100%/10% x 3 seeds"
FEATSETS="dinov3_sat student1_copy student1_copy_s1 student1_copy_s2 student2_rebuild student2_rebuild_s1 student2_rebuild_s2 student3_rebuild_only" \
  STEPS="${FINAL_STEPS:-15000}" TEMPORAL_OUT="$FINAL" bash "$D/04_temporal_runs.sh"
echo "finished L-TAE runs: $(ls "$FINAL"/*/results.json 2>/dev/null | wc -l) of $(wc -l < "$FINAL/jobs.txt")"

stage "4/4 NDVI probe with student 3, comparisons"
python scripts/ndvi_probe.py --pastis "$PASTIS" --splits "$SPLITS" --out "$RUNS/probe" \
  --feats dinov3="$TEACHER_FEATS" student1="$FEATS/student1_copy" student2="$FEATS/student2_rebuild" \
  student3="$FEATS/student3_rebuild_only" 2>&1 | tee "$RUNS/logs/probe.log"

cmp () { python scripts/summarize.py --runs "$FINAL" --compare "$1" "$2" 2>&1 | grep "^mIoU" || true; }
gap () { python scripts/summarize.py --runs "$FINAL" --compare-gap "$1" "$2" "$3" "$4" 2>&1 | grep "^mIoU" || true; }
{
for split in train train_10; do
  echo "--- labels: $split  (S1 = student1_copy*, S2 = student2_rebuild*: all 3 student seeds x 3 L-TAE seeds)"
  cmp "student2_rebuild*__ltae__real__$split" "student1_copy*__ltae__real__$split"
  cmp "student2_rebuild*__ltae__real__$split" "dinov3_sat__ltae__real__$split"
  cmp "student1_copy*__ltae__real__$split" "dinov3_sat__ltae__real__$split"
  cmp "student3_rebuild_only__ltae__real__$split" "student2_rebuild__ltae__real__$split"
  for sd in "" _s1 _s2; do      # the main result, separately for each student seed
    cmp "student2_rebuild${sd}__ltae__real__$split" "student1_copy${sd}__ltae__real__$split"
  done
  for f in "dinov3_sat" "student1_copy*" "student2_rebuild*" "student3_rebuild_only"; do
    cmp "${f}__ltae__real__$split" "${f}__ltae__shuffled__$split"
  done
  gap "student2_rebuild*__ltae__real__$split" "student2_rebuild*__ltae__shuffled__$split" \
      "student1_copy*__ltae__real__$split" "student1_copy*__ltae__shuffled__$split"
  gap "student2_rebuild*__ltae__real__$split" "student2_rebuild*__ltae__shuffled__$split" \
      "dinov3_sat__ltae__real__$split" "dinov3_sat__ltae__shuffled__$split"
done
} | tee "$RUNS/logs/comparisons_final.log"
cat "$FINAL/summary_by_student_seed.csv" 2>/dev/null || true
echo "FINAL ROUND DONE  ($(date '+%a %H:%M'))"
