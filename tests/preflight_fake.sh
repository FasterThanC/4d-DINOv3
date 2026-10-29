#!/usr/bin/env bash
# The UCloud preflight on fake data with tiny models (CPU, ~10 min). Usage: bash tests/preflight_fake.sh [dir]
set -euo pipefail
cd "$(dirname "$0")/.."
W=${1:-/tmp/4d-dino-preflight}
rm -rf "$W"; mkdir -p "$W"
python tests/make_fake_pastis.py --out "$W" --patches 60
export DATA_DIR="$W" TEACHER_MODEL="$W/tiny_teacher" STUDENT_INIT="$W/tiny_student_init"
export EXPECT=60 PILOT="36,12,12" SKIP_GPU_CHECK=1 MARGIN=0 PROBE_MARGIN=-100 PILOT_EPOCHS=30
export SPLITS="$W/splits.json"
python scripts/make_splits.py --pastis "$W/PASTIS" --out "$SPLITS" > /dev/null
bash ucloud/preflight.sh
