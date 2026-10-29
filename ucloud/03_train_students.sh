#!/usr/bin/env bash
# Train the students on the saved DINOv3 tokens (no labels), then save their tokens.
#   bash ucloud/03_train_students.sh            # student 1 and 2, one after the other
#   bash ucloud/03_train_students.sh student1   # only student 1 (copy only)
#   bash ucloud/03_train_students.sh student2   # only student 2 (copy + rebuild the 7 extra bands)
#   bash ucloud/03_train_students.sh student3   # optional control: rebuild only, no copying of DINOv3
#   bash ucloud/03_train_students.sh student2_seed1   # optional: student 2 again with seed 1
#   bash ucloud/03_train_students.sh seeds      # students 1 and 2 again with seeds 1 and 2
# Restartable: training resumes from the last epoch, finished extractions are skipped.
# Extra arguments after the name go to train_student.py, e.g.  ... student2 --epochs 30
# If you retrain a student from scratch, delete $RUNS/students/<name> AND $FEATS/<name> first.
set -euo pipefail
source "$(dirname "$0")/config.sh"
WHICH="${1:-both}"; shift || true

train_one () {   # name lambda copy_weight [extra args]
  python scripts/train_student.py --pastis "$PASTIS" --teacher "$TEACHER_FEATS" --splits "$SPLITS" \
    --out "$RUNS/students" --name "$1" --lam "$2" --copy-weight "$3" --init "$STUDENT_INIT" \
    --workers "$WORKERS" --resume "${@:4}" 2>&1 | tee -a "$RUNS/logs/train_$1.log"
  python scripts/extract_student.py --pastis "$PASTIS" --ckpt "$RUNS/students/$1/best.pt" \
    --out "$FEATS/$1" 2>&1 | tee -a "$RUNS/logs/extract_$1.log"
}

case "$WHICH" in
  student1) train_one student1_copy 0 1 "$@" ;;
  student2) train_one student2_rebuild 1 1 "$@" ;;
  student3) train_one student3_rebuild_only 1 0 "$@" ;;
  student2_seed1) train_one student2_rebuild_s1 1 1 --seed 1 "$@" ;;
  seeds)    for sd in 1 2; do
              train_one "student1_copy_s$sd" 0 1 --seed "$sd" "$@"
              train_one "student2_rebuild_s$sd" 1 1 --seed "$sd" "$@"
            done ;;
  both)     train_one student1_copy 0 1 "$@"; train_one student2_rebuild 1 1 "$@" ;;
  *) echo "usage: $0 [both|student1|student2|student3|student2_seed1|seeds] [extra args]"; exit 1 ;;
esac
