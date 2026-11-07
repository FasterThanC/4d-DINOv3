"""Experiment 1 (Time) table from runs made with scripts/train_temporal.py (our L-TAE and the
Experiment 2 training settings), in the layout of Pair 1's collect.py.

Test mIoU on fold 5, mean +- std over seeds. Key differences: seeds pooled, paired bootstrap
95% CI over test series (as in Experiment 2). The single-date month is chosen on fold 4 with
seed 0 only, so the month that later gets more seeds is not favoured.

    python scripts/exp1_table.py --runs $RUNS/exp1_ours --out $RUNS/exp1_ours/table_exp1.md --check
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
from src.metrics import bootstrap_miou  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--out", default="")
ap.add_argument("--feat", default="dinov3_sat")
ap.add_argument("--labels", default="train")
ap.add_argument("--n-boot", type=int, default=1000)
ap.add_argument("--check", action="store_true", help="exit 1 if a planned run is missing")
a = ap.parse_args()

runs = defaultdict(list)                      # setting name (without __s<seed>) -> results
for f in sorted(glob.glob(os.path.join(a.runs, "*", "results.json"))):
    r = json.load(open(f))
    runs[re.sub(r"__s\d+$", "", r["run"])].append(r)

F, L = a.feat, a.labels
key = lambda temporal, dates: f"{F}__{temporal}__{dates}__{L}"   # noqa: E731
singles = {int(m.group(1)): k for k in runs if (m := re.match(rf"^{F}__mean-cal(\d\d)__none__{L}$", k))}
s0_val = lambda k: next((r["val_miou_best"] for r in runs[k] if r["seed"] == 0), -1.0)   # noqa: E731
best = max(singles, key=lambda m: s0_val(singles[m])) if singles else None
SET = {"single": singles.get(best), "mean_all": key("mean", "none"), "mean_clear": key("mean-cloudfree", "none")}
PES = [("none", "none"), ("position", "position"), ("date", "real"), ("both", "both")]
for d, t in (("all", "ltae"), ("clear", "ltae-cloudfree")):
    for pe, dates in PES:
        SET[f"ltae_{d}_{pe}"] = key(t, dates)
SET["ltae_all_shuffled"] = key("ltae", "shuffled")

missing = [n for n, k in SET.items() if k is None or len(runs.get(k, [])) < 3]
missing += [f"single month {m}" for m in range(1, 13) if m not in singles]
if missing:
    print("MISSING or fewer than 3 seeds:", ", ".join(missing))
    if a.check:
        sys.exit(1)

# the same cloud rule must hold for every cloud-free run
for n, k in SET.items():
    if k and "cloudfree" in k:
        bad = {(r.get("max_cloud"), r.get("min_dates")) for r in runs.get(k, [])} - {(0.2, 8)}
        assert not bad, f"{k}: cloud rule {bad}, expected max_cloud 0.2, min_dates 8"


def stat(k, field="test_miou"):
    v = [r[field] for r in runs.get(k, [])]
    return (np.mean(v), np.std(v), len(v)) if v else (np.nan, np.nan, 0)


fmt = lambda s: f"{s[0]:.1f} ± {s[1]:.1f} (n={s[2]})" if s[2] else "—"   # noqa: E731


def pooled(k):
    cms = pids = None
    for r in runs[k]:
        d = np.load(os.path.join(a.runs, r["run"], "test_confusions.npz"))
        if cms is None:
            cms, pids = d["cms"].astype(np.int64), d["pids"]
        else:
            assert (d["pids"] == pids).all()
            cms = cms + d["cms"]
    return cms, pids


out = [f"# Experiment 1 (Time) in our code: {F} features, our L-TAE + decoder, Experiment 2 settings",
       "", "Test mIoU on fold 5, mean ± std over seeds (n = number of seeds).", "",
       "## Baselines (test mIoU, fold 5)"]
if best:
    out.append(f"| Single date (calendar month {best:02d}, chosen on val) | {fmt(stat(SET['single']))} |")
out += [f"| Mean of all dates | {fmt(stat(SET['mean_all']))} |",
        f"| Mean of cloud-free dates | {fmt(stat(SET['mean_clear']))} |", "",
        "## L-TAE grid (test mIoU)", "| | none | position | date | both |", "|---|---|---|---|---|"]
for d in ("all", "clear"):
    out.append(f"| {d} dates | " + " | ".join(fmt(stat(SET[f'ltae_{d}_{pe}'])) for pe, _ in PES) + " |")
out += ["", f"Shuffled dates (all): {fmt(stat(SET['ltae_all_shuffled']))}", "",
        "## Key differences (test mIoU points; seeds pooled, paired bootstrap 95% CI over test series)"]
DIFFS = [("time series vs one date", "mean_all", "single"),
         ("removing clouds (mean)", "mean_clear", "mean_all"),
         ("attention beyond cloud removal (L-TAE all, no time vs mean of cloud-free)", "ltae_all_none", "mean_clear"),
         ("order (all dates)", "ltae_all_position", "ltae_all_none"),
         ("calendar date beyond order (all)", "ltae_all_date", "ltae_all_position"),
         ("order (clear dates)", "ltae_clear_position", "ltae_clear_none"),
         ("calendar date beyond order (clear)", "ltae_clear_date", "ltae_clear_position"),
         ("real vs shuffled dates (all)", "ltae_all_date", "ltae_all_shuffled"),
         ("removing clouds (L-TAE, date)", "ltae_clear_date", "ltae_all_date"),
         ("L-TAE (all, date) vs one date", "ltae_all_date", "single")]
diffs = {}
for label, x, y in DIFFS:
    kx, ky = SET.get(x), SET.get(y)
    if not (kx in runs and ky in runs):
        continue
    (A, pa), (B, pb) = pooled(kx), pooled(ky)
    assert (pa == pb).all(), "different test series"
    ci = bootstrap_miou(A, n_boot=a.n_boot, other=B)
    diffs[label] = {"a": kx, "b": ky, "point": ci["point"], "ci95": ci["ci95"],
                    "mean_of_seed_means": stat(kx)[0] - stat(ky)[0]}
    out.append(f"- {label}: {stat(kx)[0] - stat(ky)[0]:+.1f}  (pooled {ci['point']:+.2f}, "
               f"95% CI [{ci['ci95'][0]:+.2f}, {ci['ci95'][1]:+.2f}])")
out += ["", "## Single date by calendar month, seed 0 (val / test)"]
for m in sorted(singles):
    r0 = next((r for r in runs[singles[m]] if r["seed"] == 0), None)
    if r0:
        out.append(f"  month {m:02d}: {r0['val_miou_best']:.1f} / {r0['test_miou']:.1f}")
text = "\n".join(out)
print(text)
if a.out:
    open(a.out, "w").write(text + "\n")
    json.dump({"settings": SET, "best_month": best, "differences": diffs,
               "table": {n: dict(zip(("mean", "std", "n"), map(float, stat(k)))) for n, k in SET.items() if k}},
              open(a.out.replace(".md", ".json"), "w"), indent=1)
    print("saved", a.out)
