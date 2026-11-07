#!/bin/bash
# UCloud batch job: predictions and ground-truth comparisons for every finished run.

set -euo pipefail
cd /work/ATDL_A3/exp1
export PASTIS=/work/ATDL_A3/data/PASTIS
export FEATS=/work/ATDL_A3/data/feats
export RUNS=/work/ATDL_A3/exp1/runs
export N_IDS="${N_IDS:-4}"

mapfile -t IDS < <(python - <<'PY'
import json, os
meta = json.load(open(os.path.join(os.environ["PASTIS"], "metadata.geojson")))
ids = [str(f["properties"]["ID_PATCH"]) for f in meta["features"]
       if int(f["properties"]["Fold"]) == 5]
print("\n".join(ids[: int(os.environ["N_IDS"])]))
PY
)
echo "comparing test series: ${IDS[*]}"

mkdir -p "$RUNS/comparisons"
n=0
for run in "$RUNS"/*/; do
  name=$(basename "$run")
  [[ -f "$run/best.pt" && -f "$run/result.json" ]] || continue
  if [[ ! -f "$run/preds.done" ]]; then
    python predict.py save "$run" --pastis "$PASTIS" --feat_dir "$FEATS" --workers 4
    touch "$run/preds.done"
  fi
  python predict.py compare "$run" --ids "${IDS[@]}" --pastis "$PASTIS" \
      --out "$RUNS/comparisons/${name}.png"
  n=$((n + 1))
done
echo "finished $n runs; maps in each run's preds/, pictures in $RUNS/comparisons/"
