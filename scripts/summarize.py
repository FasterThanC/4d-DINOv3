"""Collect all results.json of temporal runs into one table (mean +- std over seeds),
and give paired bootstrap 95% confidence intervals for comparisons.

    python scripts/summarize.py --runs $RUNS/temporal --out $RUNS/summary.csv

Compare two settings. Give the run name WITHOUT the "__s<seed>" ending to pool all seeds
(the per-patch confusion matrices of all seeds are added up), or with it for one seed.
A "*" pools several feature sets, e.g. all student seeds: student2_rebuild*__ltae__real__train
    python scripts/summarize.py --runs $RUNS/temporal --compare \
        student2_rebuild__ltae__real__train  student1_copy__ltae__real__train
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import bootstrap_gap_difference, bootstrap_miou  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--out", default="")
ap.add_argument("--compare", nargs=2, default=None, help="two run names (or names without __s<seed>)")
ap.add_argument("--compare-gap", nargs=4, default=None,
                metavar=("A_REAL", "A_SHUFFLED", "B_REAL", "B_SHUFFLED"),
                help="does A lose more than B when dates are shuffled? CI of (A_real-A_shuf)-(B_real-B_shuf)")
ap.add_argument("--n-boot", type=int, default=1000)
a = ap.parse_args()

groups = defaultdict(list)
for f in sorted(glob.glob(os.path.join(a.runs, "*", "results.json"))):
    r = json.load(open(f))
    key = (r["feature_name"], r["temporal"], r["dates"], r["train_split"])
    groups[key].append(r)

lines = ["features,temporal,dates,labels,n_seeds,test_miou_mean,test_miou_std,test_oa_mean,field_acc_mean"]
for key, rs in sorted(groups.items()):
    m = np.array([r["test_miou"] for r in rs])
    oa = np.array([r["test_oa"] for r in rs])
    fa = np.array([r["test_field_acc"] for r in rs])
    lines.append(",".join(map(str, key)) + f",{len(rs)},{m.mean():.2f},{m.std():.2f},{oa.mean():.2f},{fa.mean():.2f}")
# Same table with student seeds grouped (student2_rebuild, student2_rebuild_s1, ... -> student2_rebuild):
# mean and std over student seeds of the per-student mean over L-TAE seeds.
fam = defaultdict(list)
for (feat, tmp, dates, split), rs in groups.items():
    base = re.sub(r"_s\d+$", "", feat)
    fam[(base, tmp, dates, split)].append(np.mean([r["test_miou"] for r in rs]))
fam_lines = ["features,temporal,dates,labels,n_student_seeds,test_miou_mean,test_miou_std_over_student_seeds"]
for key, ms in sorted(fam.items()):
    fam_lines.append(",".join(map(str, key)) + f",{len(ms)},{np.mean(ms):.2f},{np.std(ms):.2f}")
if not (a.compare or a.compare_gap):
    print("\n".join(lines))
    if any(len(v) > 1 for v in fam.values()):
        print("\n".join(fam_lines))
if a.out:
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    if any(len(v) > 1 for v in fam.values()):
        with open(a.out.replace(".csv", "_by_student_seed.csv"), "w") as f:
            f.write("\n".join(fam_lines) + "\n")
    print("saved", a.out)


def load_pooled(name):
    """Per-patch confusion matrices of one run, or summed over all seeds of a setting."""
    exact = os.path.join(a.runs, name, "test_confusions.npz")
    if "*" in name:          # e.g. student2_rebuild*__ltae__real__train: all student seeds x all L-TAE seeds
        paths = sorted(glob.glob(os.path.join(a.runs, name + "__s*", "test_confusions.npz")))
    else:
        paths = [exact] if os.path.exists(exact) else sorted(
            glob.glob(os.path.join(a.runs, name + "__s*", "test_confusions.npz")))
    if not paths:
        raise SystemExit(f"no finished runs found for {name}")
    cms, pids = None, None
    for p in paths:
        d = np.load(p)
        if pids is None:
            cms, pids = d["cms"].astype(np.int64), d["pids"]
        else:
            assert (d["pids"] == pids).all(), "runs were tested on different patches"
            cms = cms + d["cms"]
    return cms, pids, len(paths)


if a.compare:
    A, pa, na = load_pooled(a.compare[0])
    B, pb, nb = load_pooled(a.compare[1])
    assert (pa == pb).all(), "runs were tested on different patches"
    ci = bootstrap_miou(A, n_boot=a.n_boot, other=B)
    print(f"mIoU({a.compare[0]}, {na} seed(s)) - mIoU({a.compare[1]}, {nb} seed(s)) = {ci['point']:.2f}, "
          f"95% CI [{ci['ci95'][0]:.2f}, {ci['ci95'][1]:.2f}]")

if a.compare_gap:
    loaded = [load_pooled(n) for n in a.compare_gap]
    for _, p, _ in loaded[1:]:
        assert (p == loaded[0][1]).all(), "runs were tested on different patches"
    ci = bootstrap_gap_difference(*[c for c, _, _ in loaded], n_boot=a.n_boot)
    n = a.compare_gap
    print(f"mIoU gap: ({n[0]} - {n[1]}) - ({n[2]} - {n[3]}) = {ci['point']:.2f}, "
          f"95% CI [{ci['ci95'][0]:.2f}, {ci['ci95'][1]:.2f}]")
