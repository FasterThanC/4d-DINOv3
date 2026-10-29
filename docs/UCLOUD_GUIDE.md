# Running Pair 2 on UCloud: step by step

Three UCloud jobs, all with the **Terminal** app (Ubuntu). Each one is started from
Apps → Terminal → fill in the form → Submit.

| Job | What | Machine type | Hours to set | Cost (about) |
| --- | --- | --- | --- | --- |
| A | set up Python, download PASTIS | `cpu-amd-zen5` (any size, e.g. 8 vCPU) | 3 | core-hours only, no GPU |
| B | **preflight**: everything on 180 series, with checks | `gpu-nvidia-b200`, slider at the smallest MIG (1/7) | 2 | about 0.3 GPU-hours |
| C | full run (batch job, ends by itself) | `gpu-nvidia-b200`, slider at **1** GPU | from the preflight estimate x 1.5 (e.g. 24) | about 1 GPU-hour per hour it runs |

Rule: **job C is started only after job B prints `PREFLIGHT PASSED`** and you have looked at its two
pictures. `run_all.sh` refuses to start otherwise.

The GPU hours on the dashboard (1,331 left of 7K) are shared by the whole project, so do not leave
GPU jobs open when nothing runs.

---

## 0. Before you start (once)

1. Hugging Face: open both pages and check that they say you have access (if not, accept Meta's
   terms there and wait for the approval e-mail):
   - https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m
   - https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
2. Hugging Face → Settings → Access Tokens → create a **Read** token. Keep it for job A.
3. In UCloud → Files (your Member Files, next to `DDD_PersonB`): **Create folder** `4d-dino-data`.
4. Open `4d-dino-data` and **Upload files**: `4d-dino-code.zip` (the code, in
   `upload_to_ucloud/` in your assignment folder).

The folder name matters: the scripts expect `/work/4d-dino-data`. If you use another name, type
`export DATA_DIR=/work/<name>` in every new terminal before the commands below.

## 1. Job A: setup + download (CPU)

Form: Job name `4d-setup`, Hours `3`, Machine type `cpu-amd-zen5` (8 vCPU is enough),
Folder #1 → `4d-dino-data`, Enable tmux → `On`. Submit, wait until it runs, open the terminal.

```bash
cd /work/4d-dino-data
unzip -q -o 4d-dino-code.zip -d code || python3 -m zipfile -e 4d-dino-code.zip code
cd code
bash ucloud/00_setup.sh                    # ~5 min: Python 3.12 + packages into /work/4d-dino-data
source ucloud/config.sh && hf auth login   # paste the Read token; answer "n" to the git question
bash ucloud/01_download_pastis.sh          # ~30-60 min: download 29 GB, md5 check, unzip, data check
```

`00_setup.sh` ends with `FAIL no usable GPU`: this is expected on a CPU machine.
`01_download_pastis.sh` must end with `DATA CHECK PASSED` and `PASTIS is ready`.
Then delete the zip (`rm /work/4d-dino-data/download/PASTIS.zip`) and **stop the job** (job page →
Stop), so it does not use hours.

With tmux on, closing the browser tab does not stop the commands. To get back to them, open the
job's terminal again (tmux keeps the session).

## 2. Job B: preflight (small GPU slice)

Form: Job name `4d-preflight`, Hours `2`, Machine type `gpu-nvidia-b200`, slider at the far left
(1/7, "gpu-nvidia-b200-1-mig.1g", 23 GB), Folder #1 → `4d-dino-data`, tmux `On`. Submit, open terminal:

```bash
cd /work/4d-dino-data/code
bash ucloud/preflight.sh
```

It runs 9 steps and stops at the first failed check:

1. unit tests of the code (band order, mIoU formula, L-TAE date handling, student gradients)
2. the GPU works with this PyTorch (if not, it installs a PyTorch for older drivers and checks again)
3. your Hugging Face account can download both DINOv3 models
4. PASTIS: all 2,433 series, shapes, dates = images, labels, band order (NIR > red)
5. DINOv3 tokens for 180 pilot series (120 train / 30 val / 30 test): shapes, no NaN, dates match;
   a linear probe from one token to its crop class must beat "always the most common class";
   picture `check_dinov3.png`
6. students 1 and 2 trained for 4 short epochs: cosine to DINOv3 goes up; student 2 rebuilds the
   7 extra bands better than student 1 and better than "always the band average"; student 2 still
   copies DINOv3
7. student 2 tokens: same checks as step 5; picture `check_student2.png`
8. L-TAE on the pilot: training loss goes down; test accuracy beats "always the most common class";
   shuffled dates, 10% labels, summary table and bootstrap comparison all run
9. **time estimate of the full run** on this machine type

Expected time: 20-60 min. At the end:

```
PREFLIGHT PASSED
```

Then open in the UCloud file browser `4d-dino-data/preflight/check_dinov3.png`:
the RGB pictures must look like normal fields (not black, not white, no strange colours), and the
token pictures should show the same field shapes. Write down the time estimate. **Stop the job.**

If it says `PREFLIGHT FAILED at step N`: do not start job C. Read
`4d-dino-data/runs/logs/preflight.log` (and the PNG if step 5 or 7): the FAIL line says what is wrong.

Pilot numbers (mIoU etc.) are on tiny data and are **not results**.

## 3. Job C: the full run (batch job, stops by itself)

Form: Job name `4d-full`, Machine type `gpu-nvidia-b200`, slider at **1** (one whole GPU: more
CPUs and memory, which matters more here than GPU speed), Hours = preflight estimate x 1.5
(if unsure, `24`), Folder #1 → `4d-dino-data`.
Under Parameters:
- **Batch processing** → select `4d-dino-data/code/ucloud/batch_full_run.sh`
- **Extra options** → `PARALLEL=6` (six L-TAE runs at a time on the whole GPU)

Submit. You can close the browser. The job runs all steps and ends by itself:

1. DINOv3 tokens for all ~110k images, then a check of every file
2. student 1 and student 2 (20 epochs each), then the student checks (stops here if student 2 does
   not rebuild the bands clearly better than student 1), then their tokens + checks
3. 36 L-TAE runs (3 feature sets x real/shuffled dates x 100%/10% labels x 3 seeds)
4. NDVI probe, size/speed benchmark, comparisons with 95% confidence intervals

Watch progress: Files → `4d-dino-data/runs/logs/run_all.log` (open it, refresh). The last line says
`batch job ended ... exit code 0` when everything is done.

If the job ends early (time limit, error after fixing): submit job C again with the same form.
Every step continues where it stopped (finished files and runs are skipped, students resume from
their last epoch).

## 3b. Second round (final numbers)

Same form as job C, with **Extra options** `RUN=final PARALLEL=8`. It runs `ucloud/run_final.sh`
(NDVI probe again, 2 more training seeds of each student + student 3, L-TAE with 15,000 steps on all
8 feature sets). Log: `runs/logs/run_final.log`; results in `runs/temporal_final/` and
`runs/logs/comparisons_final.log`. Before starting it, upload the new code zip and unzip it over
`code` in a short CPU terminal job: `cd /work/4d-dino-data && unzip -q -o <zip> -d code`.

## 4. Results

All in `4d-dino-data/runs/`:
- `summary_temporal.csv`: mIoU / OA / field accuracy per feature set x dates x labels (mean ± std of 3 seeds)
- `logs/comparisons.log`: the answers to the research question with 95% CIs (README: "How the
  results answer the question")
- `probe/probe_results.json`, `probe/fig_ndvi_season.png`: NDVI probe
- `students/*/log.csv`: student training curves
- `benchmark.json`: size and speed

Download them (select → Download) and put them in the report.

## 5. Notes

- Do not run two jobs that write to `4d-dino-data` at the same time.
- Disk use: PASTIS ~70 GB (40 GB after deleting the zip), DINOv3 tokens ~55 GB, each student ~20 GB,
  Python ~12 GB (`uv-cache` can be deleted after setup).
- Pair 1 needs `code/splits.json` and the DINOv3 tokens (`features/dinov3_sat`). Member Files are
  private to you; to share, copy them to a folder the whole group can access (ask the TA how groups
  share files in this project), or Pair 1 runs `bash ucloud/02_extract_teacher.sh` on their side.
- Optional extra runs (only if time and GPU hours allow), in a terminal on a GPU job:
  `bash ucloud/03_train_students.sh student3` (rebuild only) and `student2_seed1`.
