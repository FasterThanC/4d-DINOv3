"""Check the PASTIS folder before any GPU time is spent. Prints PASS/FAIL lines; exit code 1 on failure.

    python scripts/check_data.py --pastis $PASTIS [--expect 2433]

Checks: metadata (ids, folds, dates), every DATA_S2 file (shape, dtype, number of dates = metadata),
labels and parcels (shape, value range), NORM_S2_patch.json, and reflectance ranges on a sample.
Only the .npy headers are read for all files, so this takes a few minutes at most.
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pastis import (NUM_CLASSES, VOID_LABEL, load_band_norm, load_metadata, load_parcels,  # noqa: E402
                        load_s2, load_target, s2_path)

ap = argparse.ArgumentParser()
ap.add_argument("--pastis", required=True)
ap.add_argument("--expect", type=int, default=2433, help="expected number of series (0 = do not check)")
ap.add_argument("--sample", type=int, default=40, help="series whose values are read fully")
a = ap.parse_args()

ok = True


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
    ok &= bool(cond)


meta = load_metadata(a.pastis)
folds = Counter(m["fold"] for m in meta.values())
check("number of series", not a.expect or len(meta) == a.expect, f"{len(meta)}")
check("folds 1-5 all present", sorted(folds) == [1, 2, 3, 4, 5], dict(sorted(folds.items())))
nT = np.array([len(m["days"]) for m in meta.values()])
check("dates per series in a sane range", nT.min() >= 2 and nT.max() <= 100,
      f"min {nT.min()}, mean {nT.mean():.1f}, max {nT.max()}")
d_all = np.concatenate([m["days"] for m in meta.values()])
sorted_ok = all(np.all(np.diff(m["days"]) > 0) for m in meta.values())
check("dates strictly increasing within each series", sorted_ok)
check("dates within Sept 2018 - Dec 2019", d_all.min() >= 0 and d_all.max() <= 500,
      f"days {d_all.min()}..{d_all.max()}")

# every image file: header only
bad_shape, bad_T, missing, dtypes = [], [], [], Counter()
for pid, m in meta.items():
    p = s2_path(a.pastis, pid)
    if not os.path.exists(p):
        missing.append(pid)
        continue
    x = np.load(p, mmap_mode="r")
    dtypes[str(x.dtype)] += 1
    if x.ndim != 4 or x.shape[1:] != (10, 128, 128):
        bad_shape.append((pid, x.shape))
    elif x.shape[0] != len(m["days"]):
        bad_T.append((pid, x.shape[0], len(m["days"])))
check("all DATA_S2 files exist", not missing, f"missing {len(missing)}: {missing[:5]}")
check("all images have shape (T, 10, 128, 128)", not bad_shape, str(bad_shape[:3]))
check("number of images = number of metadata dates (all series)", not bad_T, str(bad_T[:3]))
check("image dtype", set(dtypes) <= {"int16", "uint16", "float32"}, dict(dtypes))

# norm file
mean, std = load_band_norm(a.pastis)
check("NORM_S2_patch.json: 10 means and stds, std > 0", len(mean) == 10 and (std > 0).all(),
      f"mean {np.round(mean).astype(int).tolist()}")

# values on a sample
rng = np.random.RandomState(0)
ids = list(meta)
sample = rng.choice(ids, size=min(a.sample, len(ids)), replace=False)
labels, refl, par_ok, tgt_shape_ok = Counter(), [], True, True
for pid in sample:
    pid = int(pid)
    y = load_target(a.pastis, pid)
    tgt_shape_ok &= y.shape == (128, 128)
    labels.update(np.unique(y, return_counts=False).tolist())
    par = load_parcels(a.pastis, pid)
    par_ok &= par.shape == (128, 128) and par.min() >= 0
    x = np.asarray(load_s2(a.pastis, pid)[:, :, ::8, ::8], dtype=np.float32)
    refl.append(np.median(x, axis=(0, 2, 3)))
check("labels are 128 x 128", tgt_shape_ok)
check("label values within 0..19", min(labels) >= 0 and max(labels) <= VOID_LABEL, sorted(labels))
check("several crop classes seen in the sample", len([c for c in labels if 0 < c < NUM_CLASSES]) >= 5)
check("parcel maps are 128 x 128, ids >= 0", par_ok)
refl = np.median(np.stack(refl), 0)
# B2..B4 (visible) usually a few hundred to ~1500; NIR (B8, index 6) well above red over vegetation
check("reflectance looks like x10000 scale", 50 < refl[0] < 3000 and 500 < refl[6] < 6000,
      f"median per band {np.round(refl).astype(int).tolist()}")
check("band order: NIR (B8) median > red (B4) median", refl[6] > refl[2],
      f"B8 {refl[6]:.0f} vs B4 {refl[2]:.0f}")

print("DATA CHECK PASSED" if ok else "DATA CHECK FAILED")
sys.exit(0 if ok else 1)
