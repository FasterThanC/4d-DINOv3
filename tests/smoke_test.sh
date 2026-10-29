#!/usr/bin/env bash
# End-to-end test of the whole Pair 2 pipeline on fake data with tiny random models.
# Runs on CPU in a few minutes. Usage: bash tests/smoke_test.sh [work_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
W=${1:-/tmp/4d-dino-smoke}
rm -rf "$W"
python tests/make_fake_pastis.py --out "$W"
P=$W/PASTIS; F=$W/feats; R=$W/runs
python scripts/make_splits.py --pastis "$P" --out "$W/splits.json"
python scripts/extract_teacher.py --pastis "$P" --out "$F/teacher" --model "$W/tiny_teacher" --batch 16
for s in "student1_copy 0 1" "student2_rebuild 1 1" "student3_rebuild_only 1 0"; do
  set -- $s
  python scripts/train_student.py --pastis "$P" --teacher "$F/teacher" --splits "$W/splits.json" \
    --out "$R/students" --name "$1" --lam "$2" --copy-weight "$3" --init "$W/tiny_student_init" \
    --epochs 2 --batch 16 --workers 0 --val-images 64
  python scripts/extract_student.py --pastis "$P" --ckpt "$R/students/$1/best.pt" --out "$F/$1" --batch 16
done
for feats in teacher student1_copy student2_rebuild; do
  for dates in real shuffled; do
    python scripts/train_temporal.py --pastis "$P" --feats "$F/$feats" --splits "$W/splits.json" \
      --out "$R/temporal" --dates "$dates" --steps 30 --eval-every 15 --batch 4 --workers 0
  done
done
python scripts/train_temporal.py --pastis "$P" --feats "$F/teacher" --splits "$W/splits.json" \
  --out "$R/temporal" --train-split train_10 --temporal mean --steps 20 --eval-every 10 --batch 2 --workers 0
python scripts/summarize.py --runs "$R/temporal" --out "$R/summary.csv" \
  --compare student2_rebuild__ltae__real__train__s0 student1_copy__ltae__real__train__s0 --n-boot 50
python scripts/summarize.py --runs "$R/temporal" --compare student2_rebuild__ltae__real__train student1_copy__ltae__real__train --n-boot 50
python scripts/summarize.py --runs "$R/temporal" --compare-gap student2_rebuild__ltae__real__train student2_rebuild__ltae__shuffled__train student1_copy__ltae__real__train student1_copy__ltae__shuffled__train --n-boot 50
python scripts/ndvi_probe.py --pastis "$P" --splits "$W/splits.json" --out "$R/probe" \
  --feats dinov3="$F/teacher" student1="$F/student1_copy" student2="$F/student2_rebuild" \
  --train-patches 0 --test-patches 0 --cloud 1.0
python scripts/benchmark.py --teacher "$W/tiny_teacher" --ckpt "$R/students/student2_rebuild/best.pt" \
  --out "$R/benchmark.json" --batch 4 --iters 2
echo "SMOKE TEST PASSED"
