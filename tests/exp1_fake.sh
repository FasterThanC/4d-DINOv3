#!/usr/bin/env bash
# Runs ucloud/run_exp1.sh end to end on fake data (CPU, tiny models, few steps).
# Usage: bash tests/exp1_fake.sh [work_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
W=${1:-/tmp/4d-dino-exp1}
rm -rf "$W"
python tests/make_fake_pastis.py --out "$W"
python scripts/make_splits.py --pastis "$W/PASTIS" --out "$W/splits.json"
python scripts/extract_teacher.py --pastis "$W/PASTIS" --out "$W/features/dinov3_sat" --model "$W/tiny_teacher" --batch 16
for d in real shuffled; do for s in 0 1 2; do        # stand-ins for the Experiment 2 runs
  python scripts/train_temporal.py --pastis "$W/PASTIS" --feats "$W/features/dinov3_sat" --splits "$W/splits.json" \
    --out "$W/runs/temporal_final" --dates $d --seed $s --steps 20 --workers 0 > /dev/null
done; done
DATA_DIR="$W" SPLITS="$W/splits.json" EXP1_STEPS=20 PARALLEL=4 WORKERS=4 bash ucloud/run_exp1.sh
test -f "$W/runs/exp1_ours/table_exp1_ours.md" && test -f "$W/runs/exp1_ours/attention/attention_exp1.txt"
echo "EXP1 FAKE TEST PASSED"
