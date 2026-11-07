"""Collect runs/*/result.json into the results tables.

  python collect.py --out runs
"""
import argparse, glob, json, re
from collections import defaultdict
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="runs"); a = ap.parse_args()
    res = defaultdict(list)
    for f in glob.glob(f"{a.out}/*/result.json"):
        r = json.load(open(f))
        res[re.sub(r"_s\d+$", "", r["name"])].append(r)

    def stat(key, split="test"):
        v = [r[split]["mIoU"] for r in res.get(key, [])]
        return (np.mean(v), np.std(v), len(v)) if v else (np.nan, np.nan, 0)

    fmt = lambda s: f"{s[0]:.1f} ± {s[1]:.1f} (n={s[2]})" if s[2] else "—"

    # single date: choose month on VALIDATION (fold 4), report its test score
    singles = [k for k in res if k.startswith("single_m")]
    # pick by seed-0 val only, so the month with extra seeds isn't favoured
    s0_val = lambda k: next((r["val"]["mIoU"] for r in res[k] if r["args"]["seed"] == 0), -1)
    best_single = max(singles, key=s0_val) if singles else None

    print("\n## Baselines (test mIoU, fold 5)")
    if best_single:
        print(f"| Single date ({best_single[-3:]}, chosen on val) | {fmt(stat(best_single))} |")
    print(f"| Mean of all dates | {fmt(stat('mean_all'))} |")
    print(f"| Mean of cloud-free dates | {fmt(stat('mean_clear'))} |")

    pes = ["none", "position", "date", "both"]
    print("\n## L-TAE grid (test mIoU)")
    print("| | " + " | ".join(pes) + " |\n|---|" + "---|" * len(pes))
    for d in ("all", "clear"):
        print(f"| {d} dates | " + " | ".join(fmt(stat(f"ltae_{d}_{p}")) for p in pes) + " |")
    if "ltae_all_shuffled" in res:
        print(f"\nShuffled dates (all): {fmt(stat('ltae_all_shuffled'))}")

    m = lambda k: stat(k)[0]
    print("\n## Key differences (test mIoU points)")
    rows = [
        ("time series vs one date", "mean_all", best_single),
        ("removing clouds (mean)", "mean_clear", "mean_all"),
        ("attention beyond cloud removal", "ltae_all_none", "mean_clear"),
        ("order (all dates)", "ltae_all_position", "ltae_all_none"),
        ("calendar date beyond order (all)", "ltae_all_date", "ltae_all_position"),
        ("order (clear dates)", "ltae_clear_position", "ltae_clear_none"),
        ("calendar date beyond order (clear)", "ltae_clear_date", "ltae_clear_position"),
    ]
    for label, x, y in rows:
        if x in res and y in res:
            print(f"- {label}: {m(x) - m(y):+.1f}")

    print("\n## Single date by month (val / test)")
    for k in sorted(singles):
        print(f"  {k}: {stat(k,'val')[0]:.1f} / {stat(k)[0]:.1f}")


if __name__ == "__main__":
    main()
