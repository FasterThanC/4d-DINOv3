#!/usr/bin/env bash
# PREFLIGHT: test everything on real data at small scale before the long run (~30-60 min on a
# 1/7 GPU slice). It stops at the first failed check. The full run starts only after
# "PREFLIGHT PASSED". Outputs: $DATA_DIR/preflight/ (look at the two PNG figures), log in
# $RUNS/logs/preflight.log.
#   bash ucloud/preflight.sh
# Settings (optional): PILOT="120,30,30" (train,val,test series)  PILOT_EPOCHS=4  PILOT_STEPS=600
set -euo pipefail
source "$(dirname "$0")/config.sh"
PF="$DATA_DIR/preflight"
PILOT="${PILOT:-120,30,30}"
PILOT_EPOCHS="${PILOT_EPOCHS:-4}"
PILOT_STEPS="${PILOT_STEPS:-600}"
EXPECT="${EXPECT:-2433}"
MARGIN="${MARGIN:-5}"            # how much the pilot L-TAE must beat "most common class" (OA points)
PROBE_MARGIN="${PROBE_MARGIN:-0}" # same for the one-image token probes (steps 5, 7)
mkdir -p "$PF"
rm -f "$PF/PASSED"
exec > >(tee -a "$RUNS/logs/preflight.log") 2>&1
step () { echo; echo "=== [$1/9] $2  ($(date +%H:%M:%S))"; }
fail () { echo; echo "PREFLIGHT FAILED at step $1. Do NOT start the full run; fix the problem first (see docs/UCLOUD_GUIDE.md)."; exit 1; }

step 1 "code checks (unit tests)"
python tests/test_units.py || fail 1

step 2 "GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
if [ "${SKIP_GPU_CHECK:-0}" = 1 ]; then echo "skipped (test mode)"
elif ! python scripts/check_gpu.py; then
  echo "Trying PyTorch built for CUDA 12.8 (older drivers)..."
  bash ucloud/00_setup.sh fix-torch || fail 2
fi

step 3 "Hugging Face access to the DINOv3 models"
python - <<PY || fail 3
import os, sys
from huggingface_hub import hf_hub_download
ok = True
for m in ("$TEACHER_MODEL", "$STUDENT_INIT"):
    if os.path.isdir(m) or m == "random":
        print("PASS local model", m); continue
    try:
        hf_hub_download(m, "config.json"); print("PASS access to", m)
    except Exception as e:
        ok = False; print("FAIL", m, "->", type(e).__name__, str(e).splitlines()[0])
        print("     Accept the terms on https://huggingface.co/" + m + " and run: hf auth login")
sys.exit(0 if ok else 1)
PY

step 4 "PASTIS files"
python scripts/check_data.py --pastis "$PASTIS" --expect "$EXPECT" || fail 4
[ -f "$SPLITS" ] || python scripts/make_splits.py --pastis "$PASTIS" --out "$SPLITS"
python scripts/make_splits.py --pastis "$PASTIS" --out "$PF/splits_full_copy.json" \
  --pilot "$PILOT" --pilot-out "$PF/splits_pilot.json" > /dev/null
cmp -s <(python -c "import json;d=json.load(open('$SPLITS'));print({k:v for k,v in d.items() if k!='_info'})") \
       <(python -c "import json;d=json.load(open('$PF/splits_full_copy.json'));print({k:v for k,v in d.items() if k!='_info'})") \
  || { echo "FAIL $SPLITS differs from a fresh one (made from other data?)"; fail 4; }
echo "PASS splits.json is the same as a freshly made one"

step 5 "DINOv3 features on the pilot series + check (probe, figure)"
rm -rf "$PF/feats"
python scripts/extract_teacher.py --pastis "$PASTIS" --out "$PF/feats/dinov3_sat" --model "$TEACHER_MODEL" \
  --ids-from "$PF/splits_pilot.json" 2>&1 | tee "$PF/extract_teacher.log" || fail 5
python scripts/check_features.py --pastis "$PASTIS" --feats "$PF/feats/dinov3_sat" \
  --splits "$PF/splits_pilot.json" --probe --margin "$PROBE_MARGIN" --figure "$PF/check_dinov3.png" || fail 5

step 6 "pilot students (student 1 and 2, $PILOT_EPOCHS epochs)"
rm -rf "$PF/students"
for s in "student1_copy 0" "student2_rebuild 1"; do
  set -- $s
  python scripts/train_student.py --pastis "$PASTIS" --teacher "$PF/feats/dinov3_sat" \
    --splits "$PF/splits_pilot.json" --out "$PF/students" --name "$1" --lam "$2" \
    --init "$STUDENT_INIT" --epochs "$PILOT_EPOCHS" --batch 64 --workers "$WORKERS" \
    --val-images 1000 || fail 6
done
python scripts/check_runs.py students --s1 "$PF/students/student1_copy" \
  --s2 "$PF/students/student2_rebuild" --factor 1.0 || fail 6

step 7 "student 2 features + check"
python scripts/extract_student.py --pastis "$PASTIS" --ckpt "$PF/students/student2_rebuild/best.pt" \
  --out "$PF/feats/student2_rebuild" --ids-from "$PF/splits_pilot.json" || fail 7
python scripts/check_features.py --pastis "$PASTIS" --feats "$PF/feats/student2_rebuild" \
  --splits "$PF/splits_pilot.json" --probe --margin "$PROBE_MARGIN" --figure "$PF/check_student2.png" || fail 7

step 8 "pilot L-TAE runs (DINOv3 real dates, student 2 shuffled dates) + checks"
rm -rf "$PF/temporal"
python scripts/train_temporal.py --pastis "$PASTIS" --feats "$PF/feats/dinov3_sat" \
  --splits "$PF/splits_pilot.json" --out "$PF/temporal" --dates real --steps "$PILOT_STEPS" \
  --eval-every $(( PILOT_STEPS / 4 )) --workers "$WORKERS" || fail 8
python scripts/check_runs.py temporal --run "$PF/temporal/dinov3_sat__ltae__real__train__s0" \
  --pastis "$PASTIS" --splits "$PF/splits_pilot.json" --margin "$MARGIN" || fail 8
python scripts/train_temporal.py --pastis "$PASTIS" --feats "$PF/feats/student2_rebuild" \
  --splits "$PF/splits_pilot.json" --out "$PF/temporal" --dates shuffled --train-split train_10 \
  --steps 100 --eval-every 50 --workers "$WORKERS" || fail 8
python scripts/summarize.py --runs "$PF/temporal" --out "$PF/summary.csv" || fail 8
python scripts/summarize.py --runs "$PF/temporal" --n-boot 200 --compare \
  dinov3_sat__ltae__real__train__s0 student2_rebuild__ltae__shuffled__train_10__s0 || fail 8

step 9 "time estimate for the full run"
FRAC=1.0
if nvidia-smi -L 2>/dev/null | grep -q MIG; then FRAC="${GPU_FRACTION:-0.143}"; fi
python scripts/estimate_time.py --preflight "$PF" --splits "$SPLITS" --pastis "$PASTIS" \
  --gpu-fraction "$FRAC" || fail 9

echo
date > "$PF/PASSED"
echo "PREFLIGHT PASSED  ($(date +%H:%M:%S))"
echo "Before the full run, open and look at:"
echo "  $PF/check_dinov3.png    (RGB should look like normal fields; token colours should follow the fields)"
echo "  $PF/check_student2.png"
echo "Pilot numbers are NOT results (tiny data). The full run: see docs/UCLOUD_GUIDE.md, part 4."
