# 4D-DINO: Pair 2 (Distillation) code

Course: Advanced Topics in Deep Learning (UCPH), final assignment. Deadline Sun Nov 8, 2026.

**Question for Pair 2:** a student that only copies DINOv3 cannot learn infrared, because DINOv3
never sees it. Does an extra loss that forces a 10-band student to rebuild the infrared bands make
its features carry the crop-growth signal?

## Final results (Oct 29, 2026)

Test fold 5, mIoU, L-TAE 15,000 steps, 3 L-TAE seeds; students = mean of 3 student trainings.
All numbers, CIs and logs: [results/](results/). Pair 1: see [docs/PAIR1_GUIDE.md](docs/PAIR1_GUIDE.md).

| Features | 100% labels, real dates | shuffled dates | 10% labels, real dates | NDVI R² |
| --- | --- | --- | --- | --- |
| DINOv3-SAT ViT-L (frozen) | 51.6 | 49.8 | 36.5 | 0.89 |
| Student 1, copy only (ViT-S) | 52.0 | 49.4 | 36.6 | 0.89 |
| Student 2, copy + rebuild 7 bands | **54.4** | 51.8 | **38.5** | 0.94 |
| Student 3, rebuild only (1 training) | 54.0 | 49.8 | 38.5 | 0.96 |

Student 2 − student 1: +2.4 mIoU (95% CI 2.0 to 2.9; +2.4 / +2.6 / +2.2 for the 3 trainings).
Student 1 − DINOv3: +0.4 (CI −0.1 to 0.9). Student 3 − student 2: −0.6 (CI −1.2 to 0.2).
The real − shuffled gap is not larger for student 2 than for student 1 (−0.04, CI −0.5 to 0.5).
Student: 22.3M parameters vs 303M, 4.7x more images per second.

## What the code does

```
PASTIS images ─► RGB, scale, x2 upsample ─► DINOv3-SAT (frozen) ─► saved tokens (T,1024,16,16)
                                                                      │  (also used by Pair 1)
PASTIS images (10 bands) ─► student ViT-S ── copy loss (cosine) ──────┘
                                         └── rebuild loss: 7 non-RGB bands   (student 2 only)
saved student tokens (T,384,16,16) ─► L-TAE (real / shuffled dates) ─► decoder ─► crop map, mIoU
saved tokens of DINOv3 and both students ─► NDVI probe (R², season figure, grouping)
```

| File | What it does |
| --- | --- |
| `src/pastis.py` | PASTIS metadata, dates (days since 2018-09-01), bands, splits, normalisation |
| `src/dino.py` | load DINOv3, RGB preparation (percentile clip, mean/std, upsample), patch tokens. Percentiles skip cloudy pixels (blue > 2000) |
| `src/student.py` | 10-band student (first layer: RGB weights copied, 7 bands start at 0), projector, rebuild decoder, losses |
| `src/temporal.py` | L-TAE (checked against the official code: max difference 1e-5), mean-pool baseline, decoder. Its first GroupNorm ignores padded dates (see below) |
| `src/data.py` | datasets for saved features (with real / shuffled / no dates) and for single images |
| `src/metrics.py` | confusion matrix, mIoU (same formula as U-TAE: 19 classes, void ignored), OA, field accuracy, bootstrap CIs (difference, and difference of real−shuffled gaps) |
| `scripts/make_splits.py` | `splits.json`: folds 1-3 / 4 / 5 and fixed 10% and 50% label subsets |
| `scripts/extract_teacher.py` | saves DINOv3 tokens for all images (restartable) |
| `scripts/train_student.py` | trains student 1 (`--lam 0`), student 2 (`--lam 1`) or student 3 (`--lam 1 --copy-weight 0`), resumable |
| `scripts/extract_student.py` | saves a trained student's tokens (same format) |
| `scripts/train_temporal.py` | L-TAE + decoder on any saved features; `--dates real|shuffled|none`, `--train-split train|train_10|train_50` |
| `scripts/ndvi_probe.py` | NDVI probe on several feature sets, with RGB and all-band controls |
| `scripts/benchmark.py` | parameters and images/second, DINOv3 vs student |
| `scripts/summarize.py` | results table (mean ± std over seeds); `--compare A B` and `--compare-gap A_REAL A_SHUF B_REAL B_SHUF` give paired bootstrap CIs (seeds pooled) |
| `ucloud/*.sh` | the steps to run on UCloud (also work on any Linux GPU machine) |
| `tests/smoke_test.sh` | whole pipeline on fake data with tiny random models, on CPU, in a few minutes |
| `scripts/check_data.py`, `check_features.py`, `check_runs.py`, `check_gpu.py` | the checks used by the preflight and the full run (PASS/FAIL lines) |
| `ucloud/preflight.sh`, `scripts/estimate_time.py` | small real-data run of everything with checks + time estimate |
| `tests/test_units.py` | small checks: band order, cloud-free percentiles, mIoU formula, L-TAE date/padding behaviour, student gradients |

## What you need to get

1. **Hugging Face account + access to the gated DINOv3 models** (accept Meta's terms on each page):
   - teacher: https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m
   - student start weights: https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
   - then create a **read token** (Hugging Face → Settings → Access Tokens).
   While access is pending, you can test everything with the open DINOv2-L instead:
   `bash ucloud/02_extract_teacher.sh dinov2`, then
   `export TEACHER_FEATS=$FEATS/dinov2_large STUDENT_INIT=random` before steps 03-05.
   These numbers are only for testing; do not report them as DINOv3 results.
2. **PASTIS** (Zenodo record 5012942, about 29 GB zipped). Do not download it to your laptop:
   `ucloud/01_download_pastis.sh` downloads it directly on UCloud.
3. **UCloud**: a CPU job for setup/download, a small B200 slice for the preflight, one B200 for
   the full run, and a folder with about 250 GB free (see the guide).

## Running on UCloud

Step-by-step guide with the exact forms to fill in: **[docs/UCLOUD_GUIDE.md](docs/UCLOUD_GUIDE.md)**.
Short version (Terminal app, folder `4d-dino-data` added to every job, code in `4d-dino-data/code`):

```bash
# Job A, CPU machine
bash ucloud/00_setup.sh                    # Python env with uv in /work/4d-dino-data (once)
source ucloud/config.sh && hf auth login   # paste your read token (once)
bash ucloud/01_download_pastis.sh          # PASTIS + data check + splits.json (once)
# Job B, smallest GPU slice: tests everything at small scale, stops at the first problem
bash ucloud/preflight.sh                   # must end with PREFLIGHT PASSED + time estimate
# Job C, 1 GPU, batch mode: ucloud/batch_full_run.sh (runs ucloud/run_all.sh, ends by itself)
```

**How we avoid wasting GPU hours on bugs.** Nothing long starts before the preflight passes
(`run_all.sh` checks for it). The preflight runs every part of the pipeline on 180 real series with
checks that fail loudly: unit tests, GPU, model access, all PASTIS files, feature files, a linear
probe that must beat "most common class", student learning checks, L-TAE learning checks, plus
pictures to look at and a time estimate. During the full run, a check runs after each stage
(all feature files, student logs) and the run stops at the first failed check.

`run_all.sh` = all DINOv3 features → student 1 and 2 (train + save tokens) → 36 L-TAE runs
(3 feature sets × real/shuffled dates × 100%/10% labels × 3 seeds) → NDVI probe, benchmark,
bootstrap comparisons. Each step can be run on its own (`ucloud/02...05`) and **every step is
restartable**: when a job ends, start a new one and run the same command again.

Useful settings: `STUDENT_ARGS="--epochs 30" bash ucloud/run_all.sh`;
`FEATSETS="student2_rebuild" SEEDS="0" PARALLEL=4 bash ucloud/04_temporal_runs.sh`.
If you retrain a student, delete both `$RUNS/students/<name>` and `$FEATS/<name>` first
(training resumes from `last.pt`, and finished feature files are skipped).
Optional extra runs if there is time: `bash ucloud/03_train_students.sh student3` (rebuild only,
no copying of DINOv3; tells us if DINOv3 is needed at all), and a second training seed of student 2
(`bash ucloud/03_train_students.sh student2_seed1`, then add `student2_rebuild_s1` to `FEATSETS`).

## Results you get

- `$RUNS/students/<name>/log.csv`: copy loss, cosine similarity and rebuild error per epoch, on
  training images and on fold 4 (places the student has not seen). Plot these for the report.
- `$RUNS/summary_temporal.csv`: test mIoU / OA / field accuracy, mean ± std over seeds, for every
  feature set × dates × labels.
- `$RUNS/logs/comparisons.log`: paired 95% CIs, e.g. student 2 − student 1, real − shuffled dates.
- `$RUNS/probe/probe_results.json` + `fig_ndvi_season.png`: NDVI R² (with RGB and all-band
  controls) and grouping scores (ARI, NMI).
- `$RUNS/benchmark.json`: parameters and speed of DINOv3 vs the student.

## How the results answer the question

| Question | Where to look | What supports "yes" |
| --- | --- | --- |
| Does a plain copy keep what DINOv3 has? | `comparisons.log`: student 1 − DINOv3 | CI includes 0 or is small |
| Does the rebuild loss add useful information? | student 2 − student 1 (100% and 10% labels) | CI above 0 |
| Do the features now carry growth (infrared) signal? | `probe_results.json`: NDVI R² of student 2 vs student 1 / DINOv3 | student 2 clearly higher, and above the RGB-mean control |
| Do the features now use *when* each image was taken? | `mIoU gap` lines: (real − shuffled) of student 2 minus (real − shuffled) of student 1 / DINOv3 | CI above 0 |
| Is the student cheaper? | `benchmark.json` | fewer parameters, more images/s |

Notes for the report:
- **Shuffled dates.** With shuffled dates the L-TAE sees the same images, but their dates are wrong.
  The model can still use *which* images are in the series (it then works like a set of images,
  we checked that L-TAE without dates gives the same output for any order). So real − shuffled
  measures how much the model gains from knowing *when* each image was taken.
- **Model size.** Students are ViT-S (about 22M parameters), DINOv3-SAT is ViT-L (about 300M).
  Student vs DINOv3 differences mix "more bands" with "smaller model". The clean comparison for
  the rebuild loss is student 2 vs student 1 (same size, same training, only λ differs).
- **Seeds.** The L-TAE runs use 3 seeds and the CIs pool them; each student is trained once
  (one seed), so the CIs do not include variation from student training. Say this in the report.
- **NDVI probe.** Compare R² against the controls: the RGB-mean control shows how much NDVI can be
  guessed from colour alone; the all-bands control is roughly the best possible.
- **Checkpoints.** Each student's best epoch is chosen by its own loss on fold 4 (no labels used).
- **L-TAE padding.** In the official L-TAE code the first GroupNorm also counts the zero-padded
  dates, so a series' output depends on the length of the other series in its batch. Our copy
  ignores padded dates (without padding it gives the same output as the official code).
  Pair 1 should use the same fix or batch size 1.

## Rough cost (estimates; check with the quick run)

| Step | GPU | Time |
| --- | --- | --- |
| DINOv3 features, ~110k images | A100/H100 | under 1 h |
| Each student, 20 epochs | A100/H100 | 1-3 h |
| Each student's features | any | minutes |
| 36 L-TAE runs (3 in parallel) | any | 3-6 h |
| NDVI probe + benchmark | any / CPU | < 1 h |

Disk: PASTIS 29 GB zip + ~40 GB unzipped, DINOv3 features ~55 GB, each student ~20 GB.

## Agreements with Pair 1

- Same `splits.json` (folds 1-3 / 4 / 5, fixed 10% and 50% subsets, seed 0).
- Same feature format: `<id>.npy` (T, D, 16, 16) float16 + `<id>_days.npy`, days since 2018-09-01,
  T in the same order as `DATA_S2`; RGB scaling recipe in `rgb_percentiles.json` / `manifest.json`.
- "Shuffled dates" = the images stay the same, each image gets another image's date (random each
  time in training, fixed per patch in testing). Pair 1 should use the same definition.
- Same mIoU formula as U-TAE (`src/metrics.py`), void class 19 ignored.
- `src/temporal.py` is our own copy of L-TAE so we do not wait for Pair 1. For final numbers, both
  pairs should use the same temporal model and decoder code.

## Test without data or GPU

```bash
pip install -r requirements.txt
python tests/test_units.py   # small checks, ends with ALL PASSED
bash tests/smoke_test.sh     # fake data + tiny random models, ~3 min on CPU, ends with SMOKE TEST PASSED
bash tests/preflight_fake.sh # the UCloud preflight on fake data, ~10 min on CPU
```
