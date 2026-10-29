"""Checks on finished training runs. Prints PASS/FAIL lines; exit code 1 on failure.

    # students (pilot or full): learning happened, rebuild loss does what it should
    python scripts/check_runs.py students --s1 $RUNS/students/student1_copy --s2 $RUNS/students/student2_rebuild
    # a temporal run (L-TAE): learning happened, better than "always the most common class"
    python scripts/check_runs.py temporal --run $RUNS/temporal/<run name> --pastis $PASTIS --splits splits.json
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ok = True


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
    ok &= bool(cond)


def read_log(run_dir):
    with open(os.path.join(run_dir, "log.csv")) as f:
        rows = list(csv.DictReader(f))
    return [{k: float(v) for k, v in r.items() if v not in ("", None)} for r in rows]


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="what", required=True)
s = sub.add_parser("students")
s.add_argument("--s1", required=True, help="run folder of student 1 (copy only)")
s.add_argument("--s2", required=True, help="run folder of student 2 (copy + rebuild)")
s.add_argument("--factor", type=float, default=0.9,
               help="student 2 rebuild error must be below factor x student 1's (pilot: 1.0)")
t = sub.add_parser("temporal")
t.add_argument("--run", required=True)
t.add_argument("--pastis", required=True)
t.add_argument("--splits", required=True)
t.add_argument("--margin", type=float, default=5.0, help="OA must beat the most common class by this")
a = ap.parse_args()

if a.what == "students":
    logs = {}
    for k, d in (("student 1", a.s1), ("student 2", a.s2)):
        log = read_log(d)
        logs[k] = log
        vals = [v for r in log for v in r.values()]
        check(f"{k}: log has epochs, no NaN", len(log) >= 1 and all(math.isfinite(v) for v in vals),
              f"{len(log)} epochs")
        check(f"{k}: best.pt written", os.path.exists(os.path.join(d, "best.pt")))
        if len(log) >= 2:
            check(f"{k}: fold-4 cosine to DINOv3 went up", log[-1]["val_cos"] > log[0]["val_cos"],
                  f"{log[0]['val_cos']:.3f} -> {log[-1]['val_cos']:.3f}")
        print(f"     {k} last epoch: val cos {log[-1]['val_cos']:.3f}, val rebuild MSE "
              f"{log[-1]['val_rebuild']:.3f}, hours {log[-1].get('hours', 0):.3f}")
    l1, l2 = logs["student 1"][-1], logs["student 2"][-1]
    # Student 2 is trained to keep the 7 non-RGB bands; student 1 only measures them with a
    # detached head. If student 2 is not clearly better at rebuilding, the rebuild loss is not working.
    check("student 2 rebuilds the 7 non-RGB bands better than student 1",
          l2["val_rebuild"] < a.factor * l1["val_rebuild"], f"{l2['val_rebuild']:.3f} vs {l1['val_rebuild']:.3f}")
    base = l2.get("val_rebuild_zero", 1.0)
    check("student 2 rebuild error below 'always predict the band average'",
          l2["val_rebuild"] < base, f"{l2['val_rebuild']:.3f} vs {base:.3f}")
    # the rebuild loss should not destroy the copy: allow some loss of cosine
    check("student 2 still copies DINOv3 (cosine not far below student 1)",
          l2["val_cos"] > l1["val_cos"] - 0.15, f"{l2['val_cos']:.3f} vs {l1['val_cos']:.3f}")

if a.what == "temporal":
    from src.pastis import NUM_CLASSES, VOID_LABEL, load_splits, load_target

    log = read_log(a.run)
    check("log has evaluations, no NaN", len(log) >= 1 and all(math.isfinite(v) for r in log for v in r.values()))
    with open(os.path.join(a.run, "results.json")) as f:
        res = json.load(f)
    train_split = res["train_split"]
    splits = load_splits(a.splits)
    counts = np.zeros(NUM_CLASSES + 1, dtype=np.int64)
    for pid in splits[train_split]:
        counts += np.bincount(load_target(a.pastis, pid).ravel(), minlength=NUM_CLASSES + 1)
    major = int(counts[:NUM_CLASSES].argmax())
    cm = np.load(os.path.join(a.run, "test_confusions.npz"))["cms"].sum(0)
    true = cm.sum(1)
    oa_major = 100.0 * true[major] / max(true.sum(), 1)
    present = (true > 0).sum()
    miou_major = 100.0 * true[major] / max(true.sum(), 1) / max(present, 1)
    check(f"test OA beats 'always the most common class' by > {a.margin} points", res["test_oa"] > oa_major + a.margin,
          f"{res['test_oa']:.1f} vs {oa_major:.1f}")
    check("test mIoU beats 'always the most common class'", res["test_miou"] > miou_major,
          f"{res['test_miou']:.2f} vs {miou_major:.2f}")
    if len(log) >= 2:
        check("training loss went down", log[-1]["loss"] < log[0]["loss"],
              f"{log[0]['loss']:.3f} -> {log[-1]['loss']:.3f}")
    print(f"     {res['run']}: test mIoU {res['test_miou']:.2f}, OA {res['test_oa']:.2f}, "
          f"hours {res['hours']:.3f}")
    _ = VOID_LABEL

print("RUN CHECK PASSED" if ok else "RUN CHECK FAILED")
sys.exit(0 if ok else 1)
