# 4D-DINO: do frozen DINOv3 features see when crops grow?

Course: Advanced Topics in Deep Learning (UCPH), final assignment, group project (2 pairs).

**Research question.** Do frozen DINOv3 features let a model recognise crops by the timing of
their growth over the season, and if not, can distillation into a 10-band student add that
information?

- **Experiment 1, Time (designed by Pair 1):** one date vs mean of all dates vs mean of cloud-free
  dates vs L-TAE, with no time / order / calendar date / both, and shuffled dates.
- **Experiment 2, Distillation (Pair 2):** DINOv3 (RGB only) distilled into a 10-band ViT-S student;
  copy only vs copy + rebuild the 7 non-RGB bands vs rebuild only.

Both experiments use **the same code and training settings**, so every number below can be compared
with every other. The DINOv3 + L-TAE runs (51.6 real dates, 49.8 shuffled) are the same runs in
Tables 1–3. **Every number of the project, with 95% CIs and how it was checked:
[results/ALL_NUMBERS.md](results/ALL_NUMBERS.md).**

## Final results (PASTIS, test fold 5, mIoU %)

All rows: L-TAE + decoder, 15,000 steps, batch 8, lr 1e-3, 3 seeds (mean ± std).

How the differences are given: the runs of each setting are pooled (confusion matrices summed) and
the 95% CI comes from a paired bootstrap over the 496 test series. This pooled difference is close
to the difference of the table means but not always equal: at most 0.03 apart in Table 1 and 0.01
in Table 2 with 100% labels, up to 0.09 with 10% labels (those runs vary more). Example: cloud-free
vs all dates for the mean, pooled +0.86, difference of the means 36.98 − 36.15 = +0.83. So
differences are given with two decimals.

**Table 1 – Experiment 1 (Time), DINOv3 features** ([results/exp1_time/](results/exp1_time/))

| Input | none | position | date | both |
| --- | --- | --- | --- | --- |
| one date (best month = July, chosen on fold 4) | 29.5 ± 0.1 | | | |
| mean of all dates | 36.2 ± 0.4 | | | |
| mean of cloud-free dates | 37.0 ± 0.1 | | | |
| L-TAE, all dates | 49.8 ± 0.2 | 51.0 ± 0.1 | **51.6 ± 0.2** | 51.4 ± 0.0 |
| L-TAE, cloud-free dates | 49.1 ± 0.1 | 49.6 ± 0.5 | 51.3 ± 0.4 | 50.6 ± 0.5 |
| L-TAE, all dates, trained with shuffled dates | | | 49.8 ± 0.4 | |

Key differences (pooled, 95% CI): L-TAE vs one date +22.08 [20.71, 23.31]; mean of all dates vs one
date +6.65 [5.63, 7.71]; L-TAE without any time input vs mean of cloud-free dates +12.81
[11.84, 13.79]; real vs shuffled dates +1.76 [1.24, 2.27]; order +1.20 [0.61, 1.77], calendar date on
top of order +0.56 [0.11, 1.01]; removing cloudy dates: mean +0.86 [0.41, 1.33], L-TAE −0.28
[−0.73, 0.20].
L-TAE attention: clear dates 1.23 vs cloudy dates 0.04 (1 = equal share), highest April–August
(peak June 2.18).

**Table 2 – Experiment 2 (Distillation)** (files at the top of [results/](results/))

| Features | 100% labels, real dates | shuffled dates | 10% labels, real dates | NDVI R² |
| --- | --- | --- | --- | --- |
| DINOv3-SAT ViT-L (frozen, RGB) | 51.6 | 49.8 | 36.5 | 0.89 |
| Student 1, copy only (ViT-S, 10 bands) | 52.0 | 49.4 | 36.6 | 0.89 |
| Student 2, copy + rebuild 7 bands | **54.4** | **51.8** | **38.5** | 0.94 |
| Student 3, rebuild only (1 training) | 54.0 | 49.8 | 38.5 | **0.96** |

Students 1–2: mean of 3 student trainings × 3 L-TAE seeds. Differences (pooled, 95% CI):
student 2 − student 1 +2.40 [1.98, 2.87] (+2.42 / +2.57 / +2.20 for the 3 trainings; 10% labels
+1.83 [1.52, 2.12]); student 1 − DINOv3 +0.38 [−0.12, 0.91]; student 3 − student 2 −0.56
[−1.20, 0.18], compared with the student 2 trained with the same seed (training 0: 54.5, not the
54.4 mean of the table). The real − shuffled gap is the same for students 2 and 1 (−0.04
[−0.52, 0.47]). Student: 22.3M parameters vs 303.1M, 4.7× more images/s.

**Table 3 – trained with real dates, dates shuffled only at test time** ([results/reliance/](results/reliance/))

mIoU of the pooled confusion matrices of all runs of a row (so "real dates" can differ by up to
0.01 from the means in Tables 1–2).

| Model | real dates | shuffled at test | drop [95% CI] |
| --- | --- | --- | --- |
| U-TAE, official pre-trained weights, raw 10 bands | 62.3 | 14.0 | 48.3 [46.5, 50.0] |
| DINOv3 + L-TAE | 51.6 | 39.6 | 12.0 [10.9, 13.1] |
| Student 1 + L-TAE | 52.0 | 38.8 | 13.2 [12.1, 14.2] |
| Student 2 + L-TAE | 54.4 | 40.4 | 13.9 [12.9, 15.1] |
| Student 3 + L-TAE | 54.0 | 27.2 | 26.7 [25.3, 28.2] |

**Robustness.** Experiment 1 was also run with Pair 1's own code (different decoder, lr 3e-3,
80 epochs, batch 16) and the same padding fix ([results/robustness_pair1_code/](results/robustness_pair1_code/)).
All settings are 0.6–2.3 mIoU higher there, with the same differences between settings (e.g. real
vs shuffled +1.8, L-TAE vs one date +22.1 in both).

**In short.** Many dates help a lot, mostly because the L-TAE learns which images to use (it
ignores cloudy ones); the actual dates add only ~1.8 mIoU. Distilling DINOv3 into a 10-band student
keeps its accuracy at 13.6× fewer parameters; the band-rebuild loss adds +2.4 mIoU and more NDVI
information, but does not make the model use the dates more.

## Where the results are

| Path | What |
| --- | --- |
| `results/ALL_NUMBERS.md` | every number, CIs, settings, per-class IoU, claims the numbers support (rebuilt from the files below by `scripts/make_numbers_reference.py`) |
| `results/final_summary.csv`, `final_summary_by_student_seed.csv`, `final_comparisons.txt` | Experiment 2 table and 95% CIs |
| `results/exp2_runs.csv` / `.json` | all 96 Experiment 2 L-TAE runs (json has per-class IoU and all arguments) |
| `results/ndvi_probe_results.json`, `fig_ndvi_season.png` | NDVI probe (R², grouping) |
| `results/benchmark.json` | parameters and speed |
| `results/student_training_log.csv`, `fig_student_training.png` | student training curves (7 trainings) |
| `results/log_run_all.txt`, `log_run_final.txt`, `round1_comparisons_5000steps.txt` | logs of the distillation jobs (round 1 = shorter L-TAE training, not used) |
| `results/exp1_time/` | Table 1 (`table_exp1.md/json` with CIs), all 47 runs (`exp1_runs.csv/json`), attention (`attention_exp1.*`, `fig_attention_exp1.png`), job log |
| `results/reliance/` | Table 3 (`date_reliance.txt/json`), job log with every run |
| `results/robustness_pair1_code/` | Experiment 1 with Pair 1's code: table, 47 runs, attention, job log |

All these files are copied or built from the UCloud run folders by `scripts/collect_results.py`
(its header lists where each file comes from).

## What the code does

```
PASTIS images ─► RGB, clip, x2 upsample ─► DINOv3-SAT ViT-L (frozen) ─► saved tokens (T,1024,16,16)
PASTIS images (10 bands) ─► student ViT-S ── copy loss (cosine to DINOv3 tokens)
                                         └── rebuild loss: 7 non-RGB bands       (students 2, 3)
saved tokens (DINOv3 or student) ─► choose dates (all / cloud-free / one per month)
                                 ─► L-TAE (no time / order / date / both / shuffled) or mean ─► decoder ─► crop map
```

| File | What it does |
| --- | --- |
| `src/pastis.py` | PASTIS metadata, dates (days since 2018-09-01), bands, splits, normalisation |
| `src/dino.py` | load DINOv3, RGB preparation (percentile clip, mean/std, upsample), patch tokens |
| `src/student.py` | 10-band student (RGB weights copied, 7 bands start at 0), projector, rebuild head, losses |
| `src/temporal.py` | L-TAE (input GroupNorm ignores padded dates), time encodings (date, position, both), mean baseline, decoder |
| `src/clouds.py` | cloud share per image (blue > 2000), date selection: all, cloud-free (≥ 8 kept), one per calendar month |
| `src/data.py` | datasets for saved features (real / shuffled / no / position dates, date selection) and for single images |
| `src/metrics.py` | confusion matrix, mIoU (U-TAE formula, void ignored), OA, field accuracy, paired bootstrap CIs |
| `scripts/extract_teacher.py`, `extract_student.py` | save DINOv3 / student tokens for all images |
| `scripts/train_student.py` | student 1 (`--lam 0`), 2 (`--lam 1`), 3 (`--lam 1 --copy-weight 0`) |
| `scripts/train_temporal.py` | L-TAE or mean + decoder on saved tokens; `--dates real|shuffled|none|position|both`, `--select all|cloudfree|calmonth --month M`, `--train-split train|train_10|train_50` (50% was not used for the results) |
| `scripts/summarize.py` | Experiment 2 table and CIs (`--compare`, `--compare-gap`) |
| `scripts/exp1_table.py`, `attention_exp1.py` | Experiment 1 table with CIs; attention vs clouds and month |
| `scripts/eval_shuffled_dates.py`, `eval_utae_pretrained.py`, `date_reliance.py` | Table 3 (no training) |
| `scripts/ndvi_probe.py`, `benchmark.py` | NDVI probe; size and speed |
| `scripts/collect_results.py` | copies the job outputs from the UCloud data folder into `results/` and builds the combined run files and the student-training figure |
| `scripts/make_numbers_reference.py` | builds `results/ALL_NUMBERS.md` from the files in `results/` |
| `ucloud/*.sh` | the jobs as run on UCloud (also on any Linux GPU machine, see "Reproduce from scratch"); `run_embeddings.sh` (`RUN=embeddings`) makes only the DINOv3 tokens; guide: [docs/UCLOUD_GUIDE.md](docs/UCLOUD_GUIDE.md) |
| `exp1_pair1_code/`, `pair1_kit/` | Pair 1's original Experiment 1 code (see its README for how it was used here) and the padding fix used for the robustness run |
| `tests/` | unit tests and end-to-end tests on fake data (CPU) |

## How to run (reproduce from scratch)

**You need:** a Hugging Face account with access to the gated models
[facebook/dinov3-vitl16-pretrain-sat493m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m) and
[facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
(and a Read token), a Linux machine with an NVIDIA GPU and internet access, and about 260 GB of disk.

**Disk (about 260 GB):** PASTIS about 40 GB unzipped (plus 29 GB for the zip during the download),
DINOv3 tokens 57 GB, 7 student token sets of about 21 GB each (115,222 images × 384 × 16 × 16,
float16), Python environment about 12 GB.

**Versions used for the results:** Python 3.12 (installed by `uv`), PyTorch 2.15.0+cu132, NVIDIA
B200 (driver 615.71.10). `requirements.txt` gives only lower bounds; the exact `transformers` and
other package versions were not recorded.

**Commands, in the order they were run.** They work on UCloud (Terminal app; there the GPU jobs
were started as batch jobs, see [docs/UCLOUD_GUIDE.md](docs/UCLOUD_GUIDE.md)) and on any Linux
GPU machine. `DATA_DIR` is the big data folder; the scripts stop if it does not exist.
`ucloud/batch_full_run.sh` runs `ucloud/run_<RUN>.sh` and writes its log to
`$DATA_DIR/runs/logs/run_<RUN>.log` (needed later by `collect_results.py`).

```bash
git clone https://github.com/FasterThanC/4d-DINOv3.git code && cd code
export DATA_DIR=/path/to/data && mkdir -p "$DATA_DIR"     # on UCloud: /work/4d-dino-data (default)
bash ucloud/00_setup.sh                       # Python + packages into $DATA_DIR      -> "Setup done"
source ucloud/config.sh && hf auth login      # paste the Read token
bash ucloud/01_download_pastis.sh             # 29 GB download + check -> "DATA CHECK PASSED", "PASTIS is ready"
bash ucloud/preflight.sh                      # everything on 180 series -> "PREFLIGHT PASSED"
bash ucloud/batch_full_run.sh PARALLEL=6                # tokens, students 1+2, round 1 -> "ALL DONE"
bash ucloud/batch_full_run.sh RUN=final PARALLEL=8      # Table 2                       -> "FINAL ROUND DONE"
bash ucloud/batch_full_run.sh RUN=reliance              # Table 3                       -> "RELIANCE DONE"
mkdir -p "$DATA_DIR/exp1" && cp -r exp1_pair1_code/. "$DATA_DIR/exp1/"   # Pair 1's code for the next job
bash ucloud/batch_full_run.sh RUN=pair1 PARALLEL=6      # robustness run                -> "PAIR1 DONE (FRESH)"
bash ucloud/batch_full_run.sh RUN=exp1 PARALLEL=8       # Table 1                       -> "EXP1 OURS DONE"
source ucloud/config.sh
python scripts/collect_results.py             # job outputs -> results/             -> "COLLECT DONE"
python scripts/make_numbers_reference.py      # rebuilds results/ALL_NUMBERS.md
```

Each log ends with `batch job ended ... with exit code 0` when the job finished. Every job is
restartable: start it again and finished runs are skipped. The reliance job also downloads the
official U-TAE code (commit 987874e) and weights (Zenodo 5172293, md5-checked).

| Job | What | Time |
| --- | --- | --- |
| setup + download | environment, PASTIS, data check (CPU is enough) | not timed; roughly 5 min setup and 30–60 min download |
| preflight | small test of every step | 12 min (1/7 of a B200) |
| run_all (`PARALLEL=6`) | DINOv3 tokens, students 1 + 2, first L-TAE round, probe, benchmark | 1 h 04 min (1 B200) |
| final (`RUN=final PARALLEL=8`) | 4 more student trainings + student 3, 96 L-TAE runs (Table 2), probe | 3 h 20 min (1 B200) |
| reliance (`RUN=reliance`) | Table 3 | 5 min |
| pair1 (`RUN=pair1 PARALLEL=6`) | robustness run with Pair 1's code | 2 h 42 min (1 B200) |
| exp1 (`RUN=exp1 PARALLEL=8`) | Table 1 | 1 h 05 min (1 B200) |

In total 8 h 12 min on one B200 for the four large jobs, plus the 12-minute preflight on a small
GPU slice and the 5-minute reliance job. To watch a job: `tail -f $DATA_DIR/runs/logs/run_<RUN>.log`
(`run_all.log` for the first one).

## Notes

- **Data.** 2,433 series, 38–61 images each (17 Sept 2018 – 27 Oct 2019), folds 1–3 train
  (1,455 series), 4 validation (482), 5 test (496); 10% label subset = 146 fixed training series.
- **Two kinds of shuffled dates.** Tables 1–2: trained and tested with shuffled dates (each image
  gets another image's date) = what correct dates are worth. Table 3: trained with real dates,
  shuffled only at test = how much a trained model relies on them.
- **L-TAE padding.** The official L-TAE normalises its input over the zero-padded dates too, so a
  series' output depends on the other series in its batch. Our L-TAE (and the robustness run)
  ignores padded dates; without padding it gives the same output as the official code.
- **Training dates.** L-TAE training uses up to 32 randomly drawn dates per series (redrawn each
  time); validation and test use all dates.
- **Limits.** One test fold; 80 m tokens; clouds from a blue threshold only; student 3 trained once;
  student vs DINOv3 mixes more bands with a smaller model (student 2 vs student 1 is the clean
  comparison); U-TAE only as official pre-trained weights.

## Test without data or GPU

```bash
pip install -r requirements.txt
python tests/test_units.py   # ends with ALL PASSED
bash tests/smoke_test.sh     # whole distillation pipeline on fake data, ~3 min on CPU
bash tests/exp1_fake.sh      # Experiment 1 job on fake data, ~15 min on CPU
bash tests/preflight_fake.sh # the UCloud preflight on fake data with tiny models, ~10 min on CPU
```
