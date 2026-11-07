"""Decide, with a rule fixed in advance, whether Experiment 1's L-TAE runs must be redone with the
padding fix. Compares the masked-norm reruns (seed 0) with the official-code runs (seeds 0-2).

Rule: redo all L-TAE runs if, for ltae_all_date or ltae_clear_date,
  |masked - official seed 0| > max(1.0, 2 x std over the 3 official seeds)   (mIoU points)
or if the two changes differ by more than 1.0 point (the change would alter all-vs-clear).
Prints REDO or KEEP and writes padding_check.json.

    python padding_check.py --official runs_official --masked runs_masked --out padding_check.json
"""
import argparse, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--official", required=True); ap.add_argument("--masked", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
test = lambda d, n: json.load(open(os.path.join(d, n, "result.json")))["test"]["mIoU"]  # noqa: E731

rows, deltas, redo = {}, {}, False
for cfg in ("ltae_all_date", "ltae_clear_date"):
    off = [test(a.official, f"{cfg}_s{s}") for s in (0, 1, 2)]
    mk = test(a.masked, f"{cfg}_s0")
    d, sd = mk - off[0], float(np.std(off))
    lim = max(1.0, 2 * sd)
    deltas[cfg] = d
    rows[cfg] = {"official_seeds": off, "official_std": sd, "masked_s0": mk, "change_s0": d, "limit": lim,
                 "too_big": abs(d) > lim}
    redo |= abs(d) > lim
    print(f"{cfg}: official s0 {off[0]:.2f} (std over seeds {sd:.2f}), masked s0 {mk:.2f}, change {d:+.2f} "
          f"(limit {lim:.2f})")
diff = deltas["ltae_clear_date"] - deltas["ltae_all_date"]
rows["change_clear_minus_change_all"] = diff
redo |= abs(diff) > 1.0
print(f"change(clear) - change(all) = {diff:+.2f} (limit 1.00)")
verdict = "REDO" if redo else "KEEP"
rows["verdict"] = verdict
json.dump(rows, open(a.out, "w"), indent=1)
print(verdict)
