"""Write splits.json: official folds (train 1-3, val 4, test 5) + fixed 10% and 50% label subsets.

Share this file with Pair 1 so both experiments use exactly the same subsets.

    python scripts/make_splits.py --pastis /work/data/PASTIS --out splits.json
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pastis import load_metadata, make_splits  # noqa: E402
from src.utils import save_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pastis", required=True)
ap.add_argument("--out", default="splits.json")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pilot", default="", help="also write a small pilot split, e.g. 120,30,30 "
                "(train, val, test series) used by ucloud/preflight.sh")
ap.add_argument("--pilot-out", default="splits_pilot.json")
a = ap.parse_args()

meta = load_metadata(a.pastis)
splits = make_splits(meta, seed=a.seed)
splits["_info"] = ("train = folds 1-3, val = fold 4, test = fold 5; train_10/train_50 are fixed "
                   f"random subsets of train (seed {a.seed}).")
if a.pilot:
    n_tr, n_va, n_te = (int(x) for x in a.pilot.split(","))
    rng = np.random.RandomState(a.seed)
    pick = lambda ids, n: sorted(int(x) for x in rng.choice(ids, size=min(n, len(ids)), replace=False))  # noqa: E731
    pilot = {"train": pick(splits["train"], n_tr), "val": pick(splits["val"], n_va),
             "test": pick(splits["test"], n_te)}
    pilot["train_10"] = pilot["train"][: max(1, len(pilot["train"]) // 10)]
    pilot["_info"] = "small pilot subset for the preflight checks only; never use for results"
    save_json(pilot, a.pilot_out)
    print(f"pilot split: {len(pilot['train'])}/{len(pilot['val'])}/{len(pilot['test'])} series ->", a.pilot_out)
save_json(splits, a.out)
n_img = {k: int(sum(len(meta[p]["days"]) for p in v)) for k, v in splits.items() if isinstance(v, list)}
for k, v in splits.items():
    if isinstance(v, list):
        print(f"{k:9s}: {len(v):5d} series, {n_img[k]:7d} images")
print("saved", a.out)
