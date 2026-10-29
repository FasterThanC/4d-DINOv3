# For Pair 1: get the code, the DINOv3 embeddings and our results

UCloud cannot share folders from the course project, so you make the same embeddings yourselves.
It takes about 1.5 hours of waiting (mostly the PASTIS download) and about 0.3 GPU-hours.
The scripts check at the end that your embeddings and splits are identical to ours.

## What you get
- `features/dinov3_sat/<id>.npy`: DINOv3-SAT tokens, shape (dates, 1024, 16, 16), float16, dates in
  the same order as `PASTIS/DATA_S2`. `<id>_days.npy`: days since 1 Sept 2018.
  `manifest.json` and `rgb_percentiles.json`: how the RGB images were prepared.
- `code/splits.json`: train = folds 1-3, val = fold 4, test = fold 5, fixed 10% and 50% subsets.
- `results/` in this repo: all our final numbers (no need to rerun them).

## Steps (same UCloud project; the folder name must be `4d-dino-data`)

0. **Hugging Face (do this first, approval can take hours):** log in, open
   https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m and accept Meta's terms. When the
   page says "You have been granted access", create a token: Settings → Access Tokens → **Read**.
1. **UCloud Files** → Member Files → **Create folder** `4d-dino-data` → open it → **Upload files**:
   the code zip (or clone the GitHub repo into `4d-dino-data/code` in step 2).
2. **Job A (CPU, no GPU hours)**: Apps → Terminal, machine `cpu-amd-zen5`, Hours 3,
   Folder #1 = `4d-dino-data`. In the terminal:
   ```bash
   cd /work/4d-dino-data
   unzip -q -o 4d-dino-code-*.zip -d code      # or: git clone <repo url> code
   cd code
   bash ucloud/00_setup.sh                       # ends with "FAIL no usable GPU": normal on CPU
   source ucloud/config.sh && hf auth login      # paste the token when asked (NOT after the command)
   setsid nohup bash ucloud/01_download_pastis.sh > $RUNS/logs/download.log 2>&1 < /dev/null &
   ```
   The web terminal resets every few minutes; `setsid nohup` keeps the download running anyway.
   Check with `tail -n 5 /work/4d-dino-data/runs/logs/download.log`. Wait until it ends with
   `DATA CHECK PASSED` and `PASTIS is ready` (30-60 min). Do not delete anything before that.
   Then `rm /work/4d-dino-data/download/PASTIS.zip` and stop the job.
3. **Job B (GPU, batch, stops by itself)**: Apps → Terminal, machine `gpu-nvidia-b200`, slider at
   **1**, Hours 3, Folder #1 = `4d-dino-data`, Parameters → Batch processing =
   `4d-dino-data/code/ucloud/batch_full_run.sh`, Extra options = `RUN=embeddings`.
   Log: `4d-dino-data/runs/logs/run_embeddings.log`. It must end with
   `PASS RGB scaling identical to Pair 2's run`, `PASS splits.json sizes identical ...` and
   `EMBEDDINGS DONE` (about 15-20 min).

## Running your experiments on the embeddings
Use `scripts/train_temporal.py` (L-TAE + decoder, best fold-4 checkpoint tested on fold 5), with the
same settings as our final numbers so the two halves of the report can be compared:
```bash
cd /work/4d-dino-data/code && source ucloud/config.sh
python scripts/train_temporal.py --pastis $PASTIS --feats $FEATS/dinov3_sat --splits $SPLITS \
  --out $RUNS/pair1 --steps 15000 --temporal mean --dates none --train-split train --seed 0
```
- Options already there: `--temporal ltae|mean`, `--dates real|shuffled|none`,
  `--train-split train|train_10|train_50`, `--seed`. Add your own (single date, mean of cloud-free
  dates) as new options in the same script.
- Many runs at once: `FEATSETS=dinov3_sat STEPS=15000 TEMPORAL_OUT=$RUNS/pair1 PARALLEL=8 bash ucloud/04_temporal_runs.sh`.
- Tables and 95% CIs: `python scripts/summarize.py --runs $RUNS/pair1` and `--compare A B`
  (see the top of `scripts/summarize.py`).
- Our L-TAE ignores padded dates in its first normalisation; the official code does not, so a
  series' output there depends on the other series in the batch (see README).
- Our DINOv3 L-TAE results (no need to redo): real dates 51.6, shuffled dates 49.8 mIoU (100% labels);
  36.5 / 33.7 (10% labels); 3 seeds, 15,000 steps. See `results/final_summary.csv`.
