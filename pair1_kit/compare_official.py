"""Compare Pair 1's original runs (official L-TAE code) with the reruns that use the padding fix.

For every setting found in both folders: test mIoU mean +- std over seeds, and the change.
Baselines without L-TAE (single date, mean) should change only by seed noise.

    python compare_official.py --official runs_official --fixed runs_masked
"""
import argparse, glob, json, os, re
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--official", required=True); ap.add_argument("--fixed", required=True)
a = ap.parse_args()


def load(d):
    out = defaultdict(list)
    for f in glob.glob(os.path.join(d, "*", "result.json")):
        r = json.load(open(f))
        out[re.sub(r"_s\d+$", "", r["name"])].append(r["test"]["mIoU"])
    return out


off, fix = load(a.official), load(a.fixed)
print(f"{'setting':28s} {'official (n)':>18s} {'with fix (n)':>18s} {'change':>8s}")
for k in sorted(set(off) & set(fix)):
    o, f = np.array(off[k]), np.array(fix[k])
    print(f"{k:28s} {o.mean():7.2f} ± {o.std():4.2f} ({len(o)}) {f.mean():7.2f} ± {f.std():4.2f} ({len(f)}) "
          f"{f.mean() - o.mean():+8.2f}")
