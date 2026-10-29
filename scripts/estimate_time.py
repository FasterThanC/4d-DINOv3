"""Estimate the time of the full run from the preflight (pilot) runs on this machine.

    python scripts/estimate_time.py --preflight $DATA_DIR/preflight --splits splits.json --pastis $PASTIS
"""
import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pastis import load_metadata, load_splits  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--preflight", required=True)
ap.add_argument("--splits", required=True)
ap.add_argument("--pastis", required=True)
ap.add_argument("--student-epochs", type=int, default=20)
ap.add_argument("--students", type=int, default=2)
ap.add_argument("--temporal-runs", type=int, default=36)
ap.add_argument("--parallel", type=int, default=3)
ap.add_argument("--steps", type=int, default=5000)
ap.add_argument("--gpu-fraction", type=float, default=1.0, help="share of a GPU this job uses (1/7 MIG = 0.143)")
a = ap.parse_args()

pf = a.preflight
meta = load_metadata(a.pastis)
full = load_splits(a.splits)
pilot = load_splits(os.path.join(pf, "splits_pilot.json"))
n_img = lambda ids: sum(len(meta[p]["days"]) for p in ids)  # noqa: E731
all_img = sum(len(m["days"]) for m in meta.values())

rows = []
# 1. teacher extraction: last "img/s" printed by extract_teacher.py
txt = open(os.path.join(pf, "extract_teacher.log")).read()
rate = float(re.findall(r"([\d.]+) img/s", txt)[-1])
rows.append(("DINOv3 features (all images)", all_img / rate / 3600))
# 2. students: hours per training image from the pilot log (includes the checks after each epoch)
with open(os.path.join(pf, "students", "student2_rebuild", "log.csv")) as f:
    log = list(csv.DictReader(f))
h_per_img = float(log[-1]["hours"]) / (len(log) * n_img(pilot["train"]))
h_student = h_per_img * n_img(full["train"]) * a.student_epochs
rows.append((f"{a.students} students x {a.student_epochs} epochs", a.students * h_student))
# student feature extraction: assume about the teacher rate (smaller model, same reading)
rows.append((f"{a.students} students' features", a.students * all_img / rate / 3600))
# 3. temporal runs: hours per step from the pilot L-TAE run on DINOv3 features
res = json.load(open(os.path.join(pf, "temporal", "dinov3_sat__ltae__real__train__s0", "results.json")))
h_step = 1.15 * res["hours"] / res["args"]["steps"]     # +15%: the full run evaluates more series
rows.append((f"{a.temporal_runs} L-TAE runs x {a.steps} steps ({a.parallel} at a time)",
             a.temporal_runs * a.steps * h_step / a.parallel))
rows.append(("NDVI probe + benchmark (rough)", 0.5))

total = sum(h for _, h in rows)
print("\nEstimated time of the full run ON THIS MACHINE TYPE (from the pilot):")
for k, h in rows:
    print(f"  {k:55s} {h:6.1f} h")
print(f"  {'TOTAL':55s} {total:6.1f} h   -> about {total * a.gpu_fraction:.1f} GPU-hours "
      f"at {a.gpu_fraction:.3g} GPU")
print("Notes: the pilot data is small and stays in memory, so the full run can be slower when it "
      "reads from disk (mostly the L-TAE runs). Give the job about 1.5x this time; every step is "
      "restartable if the job ends early.")
