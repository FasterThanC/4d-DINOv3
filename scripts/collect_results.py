"""Copies the result files of the UCloud jobs into results/ of this repository and builds the
combined files (one file per experiment instead of one folder per run). After this, run
scripts/make_numbers_reference.py to rebuild results/ALL_NUMBERS.md.

    source ucloud/config.sh
    python scripts/collect_results.py            # reads $DATA_DIR, writes results/

Where each file comes from ($R = $DATA_DIR/runs, $E = $DATA_DIR/exp1 = Pair 1's code folder):
  final_summary.csv, final_summary_by_student_seed.csv  $R/temporal_final/summary*.csv
  final_comparisons.txt                                 $R/logs/comparisons_final.log
  round1_comparisons_5000steps.txt                      $R/logs/comparisons.log
  log_run_all.txt, log_run_final.txt                    $R/logs/run_all.log, run_final.log
  ndvi_probe_results.json, fig_ndvi_season.png          $R/probe/
  benchmark.json                                        $R/benchmark.json
  exp2_runs.csv/.json                                   $R/temporal_final/*/results.json (96 runs)
  student_training_log.csv, fig_student_training.png    $R/students/*/log.csv (7 trainings)
  exp1_time/table_exp1.md/.json                         $R/exp1_ours/table_exp1_ours.md/.json
  exp1_time/attention_exp1.*, fig_attention_exp1.png    $R/exp1_ours/attention/
  exp1_time/exp1_runs.csv/.json                         $R/exp1_ours/*/results.json (47 runs)
  exp1_time/log_run_exp1.txt                            $R/logs/run_exp1.log
  reliance/date_reliance.json                           $R/reliance/date_reliance.json
  reliance/date_reliance.txt                            $R/logs/date_reliance.txt (+ a short header)
  reliance/log_run_reliance.txt                         $R/logs/run_reliance.log
  robustness_pair1_code/pair1_code_runs.csv/.json       $E/runs_final/*/result.json (47 runs)
  robustness_pair1_code/table_pair1_code.md             $E/final_table_exp1.md
  robustness_pair1_code/attention_pair1_code.txt        $E/attention_exp1.txt
  robustness_pair1_code/fig_attention_pair1_code.png    $E/runs_final/ltae_all_date_s0/attn_test.png
  robustness_pair1_code/log_run_pair1.txt               $R/logs/run_pair1.log
"""
import argparse
import csv
import glob
import json
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "/work/4d-dino-data"))
ap.add_argument("--exp1-dir", default="", help="Pair 1's code folder with runs_final/ (default: <data-dir>/exp1)")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
a = ap.parse_args()
R = os.path.join(a.data_dir, "runs")
E = a.exp1_dir or os.path.join(a.data_dir, "exp1")
missing, written = [], []


def out(*p):
    path = os.path.join(a.out, *p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def copy(src, *dst):
    if not os.path.exists(src):
        missing.append(src)
        return
    shutil.copyfile(src, out(*dst))
    written.append(os.path.join(*dst))


def write_runs(rows, base, keys):
    """rows -> base.json (all fields) and base.csv (main fields, floats rounded to 4 decimals)"""
    json.dump(rows, open(out(base + ".json"), "w"), indent=1)
    with open(out(base + ".csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(keys)
        for r in rows:
            wr.writerow([r.get(k) if not isinstance(r.get(k), float) else round(r[k], 4) for k in keys])
    written.extend([base + ".json", base + ".csv"])


# ---- Experiment 2 (distillation) ----
copy(f"{R}/temporal_final/summary.csv", "final_summary.csv")
copy(f"{R}/temporal_final/summary_by_student_seed.csv", "final_summary_by_student_seed.csv")
copy(f"{R}/logs/comparisons_final.log", "final_comparisons.txt")
copy(f"{R}/logs/comparisons.log", "round1_comparisons_5000steps.txt")
copy(f"{R}/logs/run_all.log", "log_run_all.txt")
copy(f"{R}/logs/run_final.log", "log_run_final.txt")
copy(f"{R}/probe/probe_results.json", "ndvi_probe_results.json")
copy(f"{R}/probe/fig_ndvi_season.png", "fig_ndvi_season.png")
copy(f"{R}/benchmark.json", "benchmark.json")

rows = []
for f in sorted(glob.glob(f"{R}/temporal_final/*/results.json")):
    r = json.load(open(f))
    r["labels"] = r["train_split"]
    rows.append(r)
if rows:
    if len(rows) != 96:
        print(f"WARNING: {len(rows)} Experiment 2 runs (expected 96)")
    write_runs(rows, "exp2_runs", ["run", "feature_name", "temporal", "dates", "labels", "seed", "val_miou_best",
                                   "test_miou", "test_oa", "test_field_acc", "hours"])
else:
    missing.append(f"{R}/temporal_final/*/results.json")

logs = {os.path.basename(os.path.dirname(f)): f for f in sorted(glob.glob(f"{R}/students/*/log.csv"))}
if logs:
    if len(logs) != 7:
        print(f"WARNING: {len(logs)} student trainings (expected 7)")
    srows = [{"student": name, **row} for name, f in sorted(logs.items()) for row in csv.DictReader(open(f))]
    with open(out("student_training_log.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(srows[0].keys()))
        wr.writeheader()
        wr.writerows(srows)
    written.append("student_training_log.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"student1_copy": ("tab:blue", "Student 1 (copy)"), "student2_rebuild": ("tab:orange", "Student 2 (copy + rebuild)"),
           "student3_rebuild_only": ("tab:green", "Student 3 (rebuild only)")}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    zero = None
    for name in sorted(logs):
        base = name.split("_s")[0] if name.endswith(("_s1", "_s2")) else name
        if base not in COL:
            continue
        rr = [row for row in srows if row["student"] == name]
        ep = [int(row["epoch"]) + 1 for row in rr]
        main = base == name                                   # training 0 = full colour, trainings 1-2 light
        kw = dict(color=COL[base][0], alpha=1.0 if main else 0.35, label=COL[base][1] if main else None)
        ax[0].plot(ep, [float(row["val_cos"]) for row in rr], **kw)
        ax[1].plot(ep, [float(row["val_rebuild"]) for row in rr], **kw)
        zero = float(rr[-1]["val_rebuild_zero"])
    if zero is not None:
        ax[1].axhline(zero, color="k", ls="--", lw=0.8, label="predict band average")
    ax[0].set_title("Cosine similarity to DINOv3 (fold 4)")
    ax[1].set_title("Error rebuilding the 7 non-RGB bands (fold 4, MSE)")
    ax[1].set_yscale("log")
    for x in ax:
        x.set_xlabel("epoch")
        x.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out("fig_student_training.png"), dpi=200)
    written.append("fig_student_training.png")
else:
    missing.append(f"{R}/students/*/log.csv")

# ---- Experiment 1 (time), our code ----
copy(f"{R}/exp1_ours/table_exp1_ours.md", "exp1_time", "table_exp1.md")
copy(f"{R}/exp1_ours/table_exp1_ours.json", "exp1_time", "table_exp1.json")
for f in ("attention_exp1.txt", "attention_exp1.json", "fig_attention_exp1.png"):
    copy(f"{R}/exp1_ours/attention/{f}", "exp1_time", f)
copy(f"{R}/logs/run_exp1.log", "exp1_time", "log_run_exp1.txt")
rows = []
for f in sorted(glob.glob(f"{R}/exp1_ours/*/results.json")):
    r = json.load(open(f))
    args = r["args"]
    r["select"], r["month"], r["labels"] = args.get("select", "all"), args.get("month"), r["train_split"]
    r["reused_from_exp2"] = "select" not in args                 # the 6 linked Experiment 2 runs (older code)
    rows.append(r)
if rows:
    if len(rows) != 47:
        print(f"WARNING: {len(rows)} Experiment 1 runs (expected 47)")
    write_runs(rows, os.path.join("exp1_time", "exp1_runs"), ["run", "temporal", "select", "month", "dates", "seed",
               "val_miou_best", "test_miou", "test_oa", "test_field_acc", "reused_from_exp2"])
else:
    missing.append(f"{R}/exp1_ours/*/results.json")

# ---- dates shuffled only at test time ----
copy(f"{R}/reliance/date_reliance.json", "reliance", "date_reliance.json")
copy(f"{R}/logs/run_reliance.log", "reliance", "log_run_reliance.txt")
src = f"{R}/logs/date_reliance.txt"
if os.path.exists(src):
    head = ["Test-time date shuffling (no training): models trained with real dates, tested on fold 5 with",
            "real dates and with shuffled dates (same permutation per series for every model).",
            "mIoU; drop with paired bootstrap 95% CI over test series, seeds pooled.",
            "U-TAE = official pre-trained weights (Zenodo 5172293, Fold_1 = trained on folds 1-3, val fold 4).", ""]
    body = [l.rstrip() for l in open(src).read().splitlines() if l.strip() and not l.startswith("saved ")]
    open(out("reliance", "date_reliance.txt"), "w").write("\n".join(head + body) + "\n")
    written.append(os.path.join("reliance", "date_reliance.txt"))
else:
    missing.append(src)

# ---- robustness: Experiment 1 with Pair 1's code ----
copy(f"{E}/final_table_exp1.md", "robustness_pair1_code", "table_pair1_code.md")
copy(f"{E}/attention_exp1.txt", "robustness_pair1_code", "attention_pair1_code.txt")
copy(f"{E}/runs_final/ltae_all_date_s0/attn_test.png", "robustness_pair1_code", "fig_attention_pair1_code.png")
copy(f"{R}/logs/run_pair1.log", "robustness_pair1_code", "log_run_pair1.txt")
rows = []
for f in sorted(glob.glob(f"{E}/runs_final/*/result.json")):
    d = json.load(open(f))
    ar = d["args"]
    rows.append({"run": d["name"], "agg": ar["agg"], "dates": ar["dates"], "pe": ar["pe"],
                 "month": ar["month"] if ar["agg"] == "single" else None, "seed": ar["seed"],
                 "val_miou": d["val"]["mIoU"], "test_miou": d["test"]["mIoU"], "test_oa": d["test"]["OA"],
                 "lr": ar["lr"], "epochs": ar["epochs"], "bs": ar["bs"], "wd": ar["wd"],
                 "test_iou_per_class": d["test"]["IoU_per_class"]})
if rows:
    if len(rows) != 47:
        print(f"WARNING: {len(rows)} runs of Pair 1's code (expected 47)")
    write_runs(rows, os.path.join("robustness_pair1_code", "pair1_code_runs"),
               ["run", "agg", "dates", "pe", "month", "seed", "val_miou", "test_miou", "test_oa", "lr", "epochs", "bs", "wd"])
else:
    missing.append(f"{E}/runs_final/*/result.json")

print(f"wrote {len(written)} files to {a.out}")
for m in missing:
    print("MISSING (skipped):", m)
print("COLLECT DONE" if not missing else "COLLECT DONE WITH MISSING FILES")
