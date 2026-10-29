#!/usr/bin/env bash
# Only the DINOv3 embeddings (what Pair 1 needs): all 2,433 series, then a check of every file and a
# check that the image preparation matches Pair 2's run exactly.
# Batch job: Extra options "RUN=embeddings". About 15 min on one B200 (+ model download).
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
source "$D/config.sh"
echo "##### DINOv3 embeddings for all images  ($(date '+%a %H:%M'))"
bash "$D/02_extract_teacher.sh" full
python scripts/check_features.py --pastis "$PASTIS" --feats "$TEACHER_FEATS" --splits "$SPLITS"
python - "$TEACHER_FEATS/rgb_percentiles.json" "$SPLITS" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
ok = p["low"] == [-176.0, -119.0, -371.0] and p["high"] == [2462.0, 2037.0, 1778.0]
print(("PASS" if ok else "FAIL") + " RGB scaling identical to Pair 2's run", p["low"], p["high"])
s = json.load(open(sys.argv[2]))
n = [len(s[k]) for k in ("train", "val", "test", "train_10", "train_50")]
ok2 = n == [1455, 482, 496, 146, 728]
print(("PASS" if ok2 else "FAIL") + " splits.json sizes identical to Pair 2's (train/val/test/10%/50%)", n)
sys.exit(0 if ok and ok2 else 1)
PY
echo "EMBEDDINGS DONE  ($(date '+%a %H:%M'))"
