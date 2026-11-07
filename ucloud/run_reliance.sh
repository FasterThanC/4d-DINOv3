#!/usr/bin/env bash
# Test-time date shuffling (no training): how much does each model trained with REAL dates rely on
# them? Models: the official pre-trained U-TAE (raw 10 bands, the "from scratch" reference of the
# plan) and our final L-TAE runs on DINOv3 / student 1 / student 2 / student 3 features.
# Output: $RUNS/reliance/date_reliance.json and $RUNS/logs/date_reliance.txt
# Batch job: Extra options "RUN=reliance". Small: the smallest GPU slice (1/7) is enough.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
source "$D/config.sh"
stage () { echo; echo "##### $1  ($(date '+%a %H:%M'))"; }
OUT="$RUNS/reliance"
UT="$DATA_DIR/third_party/utae-paps"
W="$DATA_DIR/utae_weights"
mkdir -p "$OUT" "$DATA_DIR/third_party"

stage "1/4 official U-TAE code and pre-trained weights"
if [ ! -d "$UT/.git" ]; then
  git clone -q https://github.com/VSainteuf/utae-paps.git "$UT"
  git -C "$UT" checkout -q 987874e27a98bb399277f2d55eb010744d26f881
fi
echo "U-TAE code at commit $(git -C "$UT" rev-parse --short HEAD)"
if [ ! -f "$W/.done" ]; then
  mkdir -p "$W"
  curl -L -C - -o "$W/weights.zip" "https://zenodo.org/records/5172293/files/UTAE_SemanticSeg_weights.zip?download=1"
  [ "$(md5sum "$W/weights.zip" | cut -d' ' -f1)" = "91bf47d219927b2a686ec662132f754f" ] \
    || { echo "md5 of the U-TAE weights does not match"; exit 1; }
  if command -v unzip >/dev/null 2>&1; then unzip -q -o "$W/weights.zip" -d "$W"; else python -m zipfile -e "$W/weights.zip" "$W"; fi
  touch "$W/.done"
fi
find "$W" -name conf.json | head -3; find "$W" -path "*Fold_1*" -name "*.tar" | head -3

stage "2/4 U-TAE pre-trained: real vs shuffled dates on fold 5"
python scripts/eval_utae_pretrained.py --utae-code "$UT" --weights "$W" --pastis "$PASTIS" \
  --splits "$SPLITS" --out "$OUT" --min-real-miou "${MIN_UTAE_MIOU:-50}"

stage "3/4 our L-TAE runs (real dates, 100% labels): real vs shuffled dates at test"
python scripts/eval_shuffled_dates.py --pastis "$PASTIS" --splits "$SPLITS" --feats-root "$FEATS" \
  --out "$OUT" --workers "$WORKERS" --runs $(ls -d "$RUNS"/temporal_final/*__ltae__real__train__s*)

stage "4/4 table"
python scripts/date_reliance.py --dir "$OUT" --out "$OUT/date_reliance.json" | tee "$RUNS/logs/date_reliance.txt"
echo "RELIANCE DONE  ($(date '+%a %H:%M'))"
