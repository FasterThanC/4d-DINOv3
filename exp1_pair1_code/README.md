# Experiment 1: clouds × timing on frozen DINOv3-SAT features

## Files
| File | What it does |
|---|---|
| `compute_clouds.py` | Cloud fraction per date from the raw blue band (B2 > 2000). Run once. |
| `data.py` | Loads saved features + dates + clouds + labels; picks the dates each run sees; pads batches. |
| `models.py` | Shared input projection + decoder; aggregator = single / mean / L-TAE with PE switch. |
| `train.py` | One run: train folds 1–3, best epoch on fold 4, test on fold 5 → `runs/<name>/result.json`. |
| `run_grid.py` | The full grid, spread over GPUs and machines. Skips finished runs. |
| `collect.py` | Results tables + the key differences. |
| `predict.py` | Predicted crop maps from a trained run (`save`), and a picture comparing runs (`compare`). |
| `tune_grid.py` | Learning rate × epochs sweep for `ltae_all_date`, seed 0, on fold 4 (step 3). |
| `run_all.sh`, `predict_all.sh` | The UCloud jobs we used (whole grid with lr 3e-3 and 80 epochs; prediction maps). They contain fixed paths under `/work/ATDL_A3/`: change them before use. |
| `src/backbones/ltae.py`, `positional_encoding.py` | Official L-TAE (`LTAE2d`) from github.com/VSainteuf/utae-paps (MIT), unchanged in this folder. |
| `attention_analysis.py` | Attention vs cloud fraction and vs month, from a run's `attn_test.json`. |

## Steps
```bash
# run everything from inside exp1/ (src/backbones/ holds the official L-TAE files)
cd exp1

# 0. features dir must have <id>.npy and <id>_days.npy for all 2433 series
# 1. cloud fractions (also checks date counts match)
python compute_clouds.py --pastis /data/PASTIS --feat_dir /data/feats --check

# 2. smoke test: a few epochs, check val mIoU goes up
python train.py --pastis /data/PASTIS --feat_dir /data/feats --out smoke \
    --agg ltae --dates all --pe date --epochs 3

# 3. tune lr / epochs ONCE on fold 4 with ltae_all_date, then freeze them for every run

# 4. full grid (43 runs; then 2 more with --single_best = 45), e.g. 3 machines
python run_grid.py --pastis /data/PASTIS --feat_dir /data/feats --shard 0 --nshards 3   # machine 0
python run_grid.py --pastis /data/PASTIS --feat_dir /data/feats --shard 1 --nshards 3   # machine 1
python run_grid.py --pastis /data/PASTIS --feat_dir /data/feats --shard 2 --nshards 3   # machine 2

# 5. results (copy all runs/ dirs to one place first)
python collect.py --out runs
python attention_analysis.py runs/ltae_all_date_s0/attn_test.json
```

## Fixed choices (put these in the report)
- Cloud pixel: blue (B2) > 2000. Cloudy date: > 20 % cloud pixels (`--cloudy_frac`).
  Check the printed distribution from step 1 before fixing it.
- Cloud-free runs keep at least 8 dates (the least cloudy) if fewer are clear (`--min_dates`).
- `position` = renumbered order of the dates the model sees (0, 1, 2, …) × 10 days,
  so after cloud removal it carries order only, not timing.
- `date` = days since 2018-09-01. `both` = half the encoding each.
- `shuffled` = real dates randomly permuted, a fresh permutation every time a sample is loaded.
- Single-date baseline: least cloudy date in a month; the month is chosen on fold 4.
- mIoU over classes 0–18 (void = 19 ignored), as in U-TAE.

## Grid
Seed 0: 12 single-date months. Per seed (0, 1, 2): mean all, mean clear,
L-TAE {all, clear} × {none, position, date, both}. Plus shuffled dates, seed 0.
Then `run_grid.py --single_best`: the best month on fold 4 with seeds 1 and 2.
In total 12 + 3 × 10 + 1 + 2 = 45 runs.

## How this code is used in this repository
The final Experiment 1 numbers (Table 1 in the main README) come from `ucloud/run_exp1.sh`, which
runs this design with the shared L-TAE code. This folder is used only for the robustness run
(`ucloud/run_pair1.sh`, batch `RUN=pair1`):
- the folder is copied to `$DATA_DIR/exp1` (`mkdir -p $DATA_DIR/exp1 && cp -r exp1_pair1_code/. $DATA_DIR/exp1/`),
  so the commands above that say `exp1/` mean that copy;
- `run_pair1.sh` copies `pair1_kit/ltae.py` over `src/backbones/ltae.py` (the original is kept as
  `ltae_official_backup.py`) and turns on its fix: the first GroupNorm ignores padded dates;
- it runs the same 45 runs with lr 3e-3, 80 epochs, batch 16, plus shuffled dates with seeds 1 and 2
  (47 runs).
