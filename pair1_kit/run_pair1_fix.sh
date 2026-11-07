#!/usr/bin/env bash
# Experiment 1 (Pair 1 code): padding check of the official L-TAE, then either
#   KEEP: official grid stays; add shuffled-dates seeds 1 and 2 (official code)
#   REDO: all L-TAE runs again with the padding fix (single/mean runs do not use L-TAE: kept)
# and build the final table from one folder (runs_final).
# Needs: EXP1 = Pair 1's code folder (train.py, data.py, models.py, src/backbones/, collect.py)
#        with this kit's ltae.py copied over src/backbones/ltae.py,
#        $EXP1/runs_official = Pair 1's finished runs (at least every */result.json; attn_test.json).
# Batch job (1 GPU): Extra options "RUN=pair1"   (ucloud/run_pair1.sh calls this script)
set -euo pipefail
: "${EXP1:?set EXP1}" "${PASTIS:?set PASTIS}" "${FEAT:?set FEAT (DINOv3 feature folder)}"
P="${PARALLEL:-6}"; W="${WORKERS_PER_RUN:-4}"
cd "$EXP1"
OFF=runs_official; MASK=runs_masked; FIN=runs_final
# FRESH mode (Pair 1's results not available): run the whole grid with the padding fix, using
# Pair 1's settings (lr/epochs from their run_all.sh, the rest train.py defaults).
if [ -f "$MASK/.fresh_mode" ] || [ ! -f "$OFF/ltae_all_date_s0/result.json" ]; then MODE=fresh; else MODE=check; fi
mkdir -p "$MASK"; [ "$MODE" = fresh ] && touch "$MASK/.fresh_mode"     # stays fresh once started
echo "mode: $MODE"
if [ "$MODE" = fresh ] && [ -z "${HP:-}" ]; then
  HP="--lr 0.003 --epochs 80 --bs 16 --wd 0.01 --cloudy_frac 0.2 --min_dates 8 --pos_scale 10.0"
fi
# Settings: exactly those of Pair 1's finished runs (read from their result.json files; all must agree)
if [ -z "${HP:-}" ]; then
  HP=$(python - "$OFF" <<'PY'
import glob, json, sys
keys = ["lr", "epochs", "bs", "wd", "cloudy_frac", "min_dates", "pos_scale"]
vals = {k: set() for k in keys}
runs = glob.glob(sys.argv[1] + "/*/result.json")
for f in runs:
    a = json.load(open(f))["args"]
    for k in keys:
        vals[k].add(a[k])
bad = {k: v for k, v in vals.items() if len(v) != 1}
if bad:
    sys.exit(f"Pair 1's runs do not all use the same settings: {bad}")
print(" ".join(f"--{k} {next(iter(vals[k]))}" for k in keys))
PY
)
fi
echo "settings used for every new run: $HP"
mkdir -p "$MASK/logs"
T="python train.py --pastis $PASTIS --feat_dir $FEAT --workers $W $HP"
runp () { xargs -P "$P" -I{} bash -c "{}" < "$1" || echo "WARNING: some runs failed (see logs)"; }

echo "##### 1/4 cloud fractions (written next to the features, as Pair 1's data.py expects)"
ls "$FEAT"/*_cloud.npy >/dev/null 2>&1 || python compute_clouds.py --pastis "$PASTIS" --feat_dir "$FEAT" --check

if [ "$MODE" = check ]; then
echo "##### 2/4 padding check: L-TAE with masked norm, seed 0"
  : > "$MASK/jobs_check.txt"
  for d in all clear; do
    echo "LTAE_MASKED_NORM=1 $T --out $MASK --agg ltae --dates $d --pe date --seed 0 --save_attn > $MASK/logs/ltae_${d}_date_s0.log 2>&1" >> "$MASK/jobs_check.txt"
  done
  runp "$MASK/jobs_check.txt"
  python padding_check.py --official "$OFF" --masked "$MASK" --out "$MASK/padding_check.json" | tee "$MASK/padding_check.txt"
  VERDICT=$(tail -1 "$MASK/padding_check.txt")
else
  echo "##### 2/4 whole grid with the padding fix (Pair 1's own results: compare later with compare_official.py)"
  VERDICT=FRESH; OFF="$MASK"
fi
echo "##### 3/4 verdict $VERDICT: complete the grid (finished runs are skipped)"
: > "$MASK/jobs_main.txt"
# baselines (no L-TAE inside, so the padding fix does not matter): official folder
for s in 0 1 2; do for d in all clear; do
  echo "$T --out $OFF --agg mean --dates $d --seed $s > $MASK/logs/mean_${d}_s$s.log 2>&1" >> "$MASK/jobs_main.txt"
done; done
for m in $(seq 1 12); do
  echo "$T --out $OFF --agg single --month $m --seed 0 > $MASK/logs/single_m${m}_s0.log 2>&1" >> "$MASK/jobs_main.txt"
done
# L-TAE grid: with the fix (REDO) or the official code (KEEP)
if [ "$VERDICT" = "KEEP" ]; then LT="$OFF"; NORM=0; else LT="$MASK"; NORM=1; fi
for s in 0 1 2; do
  for d in all clear; do for pe in none position date both; do
    echo "LTAE_MASKED_NORM=$NORM $T --out $LT --agg ltae --dates $d --pe $pe --seed $s --save_attn > $MASK/logs/ltae${NORM}_${d}_${pe}_s$s.log 2>&1" >> "$MASK/jobs_main.txt"
  done; done
  echo "LTAE_MASKED_NORM=$NORM $T --out $LT --agg ltae --dates all --pe shuffled --seed $s > $MASK/logs/ltae${NORM}_all_shuffled_s$s.log 2>&1" >> "$MASK/jobs_main.txt"
done
runp "$MASK/jobs_main.txt"
# one date: best month on fold 4 (seed 0, as in Pair 1's collect.py), then seeds 1 and 2
BEST=$(python - "$OFF" <<'PY'
import glob, json, sys
r = [json.load(open(f)) for f in glob.glob(sys.argv[1] + "/single_m*_s0/result.json")]
assert len(r) == 12, f"only {len(r)} of 12 single-date months finished"
b = max(r, key=lambda x: x["val"]["mIoU"])
print(b["args"]["month"])
PY
)
echo "best single-date month on fold 4: $BEST"
: > "$MASK/jobs_best.txt"
for s in 1 2; do
  echo "$T --out $OFF --agg single --month $BEST --seed $s > $MASK/logs/single_best_s$s.log 2>&1" >> "$MASK/jobs_best.txt"
done
runp "$MASK/jobs_best.txt"
SRC_LTAE="$LT"

echo "##### 4/4 final table (runs_final: single/mean from $OFF, L-TAE from $SRC_LTAE)"
rm -rf "$FIN"; mkdir -p "$FIN"
for r in "$OFF"/single_* "$OFF"/mean_* "$SRC_LTAE"/ltae_*; do
  [ -f "$r/result.json" ] && ln -s "$(cd "$r" && pwd)" "$FIN/$(basename "$r")"
done
echo "runs in the final table: $(ls "$FIN" | wc -l)"
python - "$FIN" <<'PY'
import os, sys
d = sys.argv[1]
need = [f"mean_{x}_s{s}" for x in ("all", "clear") for s in (0, 1, 2)]
need += [f"ltae_{x}_{pe}_s{s}" for x in ("all", "clear") for pe in ("none", "position", "date", "both") for s in (0, 1, 2)]
need += [f"ltae_all_shuffled_s{s}" for s in (0, 1, 2)]
miss = [n for n in need if not os.path.exists(os.path.join(d, n, "result.json"))]
singles = [n for n in os.listdir(d) if n.startswith("single_m") and n.endswith("_s0")]
if len(singles) != 12:
    miss.append(f"single-date months with seed 0: {len(singles)} of 12")
best_seeds = {n.rsplit("_", 1)[0] for n in os.listdir(d) if n.startswith("single_m") and n.endswith(("_s1", "_s2"))}
if not best_seeds:
    miss.append("single-date best month, seeds 1-2")
if miss:
    print("FAIL incomplete, missing:", ", ".join(miss), "\nStart the job again: finished runs are skipped.")
    sys.exit(1)
print("PASS all", len(need) + len(singles), "+ best-month runs present")
PY
python collect.py --out "$FIN" | tee "$FIN/../final_table_exp1.md"
[ -f "$FIN/ltae_all_date_s0/attn_test.json" ] && python attention_analysis.py "$FIN/ltae_all_date_s0/attn_test.json" | tee "$FIN/../attention_exp1.txt" || true
echo "PAIR1 DONE ($VERDICT)"
