#!/usr/bin/env bash
# For a UCloud *batch* job: select this file in the Terminal app's "Batch processing" field.
# The job ends by itself when the run finishes, so no GPU hours are used after that.
# Settings can be given in the "Extra options" field as KEY=VALUE, e.g.:   PARALLEL=6
# Second round:  RUN=final PARALLEL=8   (runs ucloud/run_final.sh); Experiment 1 in our code: RUN=exp1 PARALLEL=8
# Log: /work/4d-dino-data/runs/logs/run_all.log or run_final.log (open it in the UCloud file browser)
for kv in "$@"; do
  if [[ "$kv" =~ ^[A-Z_][A-Z0-9_]*=.*$ ]]; then export "$kv"; fi
done
CODE_DIR="${CODE_DIR:-/work/4d-dino-data/code}"
if [ ! -f "$CODE_DIR/ucloud/run_all.sh" ]; then
  CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi
export CODE_DIR
export DATA_DIR="${DATA_DIR:-/work/4d-dino-data}"
mkdir -p "$DATA_DIR/runs/logs"
LOG="$DATA_DIR/runs/logs/run_${RUN:-all}.log"     # RUN=final -> run_final.sh, log run_final.log
echo "=== batch job started $(date), settings: $*" >> "$LOG"
bash "$CODE_DIR/ucloud/run_${RUN:-all}.sh" >> "$LOG" 2>&1
code=$?
echo "=== batch job ended $(date) with exit code $code (0 = everything finished)" >> "$LOG"
exit $code
