#!/usr/bin/env bash
# Experiment 1 (Time) with OUR code and the Experiment 2 training settings, so every number is
# comparable with the distillation table. Design = Pair 1's grid (their exp1/run_grid.py):
#   one date: least cloudy image of each calendar month 1-12 (seed 0), best month on fold 4 -> seeds 1, 2
#   mean of all dates, mean of cloud-free dates                                   (3 seeds each)
#   L-TAE {all, cloud-free dates} x {none, position, date, both}                   (3 seeds each)
#   L-TAE all dates, shuffled dates                                                (3 seeds)
# Cloud-free = at most 20% of pixels with blue > 2000, at least the 8 least cloudy dates kept.
# Settings of every run = Experiment 2 final round: 15,000 steps, batch 8, lr 1e-3, wd 1e-4,
# 32 random dates per series in training, best fold-4 checkpoint tested on fold 5.
# "L-TAE all dates, date" and "shuffled" ARE the Experiment 2 DINOv3 runs (linked, not retrained).
# Restartable: finished runs are skipped. Batch job: Extra options "RUN=exp1 PARALLEL=8".
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
source "$D/config.sh"
stage () { echo; echo "##### $1  ($(date '+%a %H:%M'))"; }
OUT="$RUNS/exp1_ours"
CL="$FEATS/cloud_fractions.npz"
F=dinov3_sat
PARALLEL="${PARALLEL:-8}"
TW=$(( WORKERS / PARALLEL > 6 ? 6 : (WORKERS / PARALLEL > 0 ? WORKERS / PARALLEL : 1) ))
STEPS="${EXP1_STEPS:-15000}"
mkdir -p "$OUT"

stage "1/5 cloud share of every image (blue > 2000), once"
python - <<PYEND
from src.clouds import cloud_fractions
from src.pastis import load_metadata
import numpy as np
meta = load_metadata("$PASTIS")
c = cloud_fractions("$PASTIS", meta, "$CL")
allc = np.concatenate([c[p] for p in meta])
print(f"{len(c)} series, {len(allc)} images, cloudy (>20% pixels): {100 * (allc > 0.2).mean():.1f}%")
PYEND

stage "2/5 link the Experiment 2 DINOv3 runs (L-TAE, all dates, real and shuffled dates)"
for d in real shuffled; do for s in 0 1 2; do
  src="$RUNS/temporal_final/${F}__ltae__${d}__train__s$s"
  for f in results.json best.pt test_confusions.npz; do
    [ -f "$src/$f" ] || { echo "FAIL missing $src/$f"; exit 1; }
  done
  ln -sfn "$src" "$OUT/"
done; done
python - <<PYEND
import json, glob
want = {"steps": $STEPS, "batch": 8, "lr": 0.001, "wd": 0.0001, "max_dates": 32, "eval_every": 500}
for f in sorted(glob.glob("$OUT/${F}__ltae__*__train__s*/results.json")):
    a = json.load(open(f))["args"]
    bad = {k: a.get(k) for k in want if a.get(k) != want[k]}
    assert not bad, f"{f}: settings differ from this job: {bad}"
print("linked runs use the same settings:", want)
PYEND

C="python scripts/train_temporal.py --pastis $PASTIS --feats $FEATS/$F --splits $SPLITS --out $OUT \
--train-split train --steps $STEPS --workers $TW --clouds $CL"
CF="--select cloudfree --max-cloud 0.2 --min-dates 8"
LOGD="$OUT/logs"; mkdir -p "$LOGD"
jobs="$OUT/jobs.txt"; : > "$jobs"
for m in 1 2 3 4 5 6 7 8 9 10 11 12; do
  echo "$C --temporal mean --dates none --select calmonth --month $m --seed 0 > $LOGD/cal${m}_s0.txt 2>&1" >> "$jobs"
done
for s in 0 1 2; do
  echo "$C --temporal mean --dates none --seed $s > $LOGD/mean_all_s$s.txt 2>&1" >> "$jobs"
  echo "$C --temporal mean --dates none $CF --seed $s > $LOGD/mean_clear_s$s.txt 2>&1" >> "$jobs"
  for d in none position both; do
    echo "$C --temporal ltae --dates $d --seed $s > $LOGD/ltae_all_${d}_s$s.txt 2>&1" >> "$jobs"
  done
  for d in none position real both; do
    echo "$C --temporal ltae --dates $d $CF --seed $s > $LOGD/ltae_clear_${d}_s$s.txt 2>&1" >> "$jobs"
  done
done
stage "3/5 $(wc -l < "$jobs") runs, $PARALLEL at a time"
xargs -P "$PARALLEL" -I{} bash -c "{}" < "$jobs" || echo "WARNING: some runs failed; see $LOGD. Start the job again to retry them."

stage "4/5 best calendar month on fold 4 (seed 0) -> seeds 1 and 2"
BEST=$(python - <<PYEND
import glob, json, sys
s = {}
for f in glob.glob("$OUT/${F}__mean-cal*__none__train__s0/results.json"):
    r = json.load(open(f)); s[r["month"]] = r["val_miou_best"]
if len(s) < 12:
    sys.exit(f"only {len(s)} of 12 months finished")
m = max(s, key=s.get)
print(m)
print(f"best month on fold 4: {m} ({s[m]:.2f} val mIoU)", file=sys.stderr)
PYEND
)
: > "$OUT/jobs_best.txt"
for s in 1 2; do
  echo "$C --temporal mean --dates none --select calmonth --month $BEST --seed $s > $LOGD/cal${BEST}_s$s.txt 2>&1" >> "$OUT/jobs_best.txt"
done
xargs -P 2 -I{} bash -c "{}" < "$OUT/jobs_best.txt" || echo "WARNING: a best-month run failed"

stage "5/5 table, differences with 95% CIs, attention"
python scripts/exp1_table.py --runs "$OUT" --out "$OUT/table_exp1_ours.md" --check
python scripts/attention_exp1.py --pastis "$PASTIS" --splits "$SPLITS" --feats-root "$FEATS" --clouds "$CL" \
  --out "$OUT/attention" --runs "$OUT"/${F}__ltae__real__train__s*
echo "EXP1 OURS DONE  ($(date '+%a %H:%M'))"
