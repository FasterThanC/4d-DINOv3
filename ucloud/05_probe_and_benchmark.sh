#!/usr/bin/env bash
# NDVI probe on DINOv3 and both students, speed/size benchmark, and the key comparisons
# with bootstrap 95% confidence intervals.
set -euo pipefail
source "$(dirname "$0")/config.sh"

python scripts/ndvi_probe.py --pastis "$PASTIS" --splits "$SPLITS" --out "$RUNS/probe" \
  --feats dinov3="$TEACHER_FEATS" student1="$FEATS/student1_copy" student2="$FEATS/student2_rebuild" \
  2>&1 | tee "$RUNS/logs/probe.log"

python scripts/benchmark.py --teacher "$TEACHER_MODEL" \
  --ckpt "$RUNS/students/student2_rebuild/best.pt" --out "$RUNS/benchmark.json" \
  2>&1 | tee "$RUNS/logs/benchmark.log"

# Paired bootstrap CIs, all seeds pooled. Main questions:
#   student 2 vs student 1 (does the rebuild loss help?), student 1 vs DINOv3,
#   real vs shuffled dates for each feature set (does the model use the timing?).
T="$RUNS/temporal"
cmp () { python scripts/summarize.py --runs "$T" --compare "$1" "$2" 2>/dev/null || true; }
for split in train train_10; do
  cmp "student2_rebuild__ltae__real__${split}" "student1_copy__ltae__real__${split}"
  cmp "student2_rebuild__ltae__real__${split}" "dinov3_sat__ltae__real__${split}"
  cmp "student1_copy__ltae__real__${split}" "dinov3_sat__ltae__real__${split}"
  for f in dinov3_sat student1_copy student2_rebuild; do
    cmp "${f}__ltae__real__${split}" "${f}__ltae__shuffled__${split}"
  done
  # Key test of the hypothesis: does student 2 lose MORE than student 1 / DINOv3 when dates are shuffled?
  for other in student1_copy dinov3_sat; do
    python scripts/summarize.py --runs "$T" --compare-gap \
      "student2_rebuild__ltae__real__${split}" "student2_rebuild__ltae__shuffled__${split}" \
      "${other}__ltae__real__${split}" "${other}__ltae__shuffled__${split}" 2>/dev/null || true
  done
done 2>&1 | { grep "^mIoU" || true; } | tee "$RUNS/logs/comparisons.log"
