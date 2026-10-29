"""PASTIS helpers: metadata, dates, splits, normalisation and raw file access.

PASTIS layout (after unzipping PASTIS.zip):
    PASTIS/
      metadata.geojson            one feature per patch: ID_PATCH, Fold, dates-S2, TILE, ...
      NORM_S2_patch.json          per-fold band means/stds ("Fold_1": {"mean": [...], "std": [...]})
      DATA_S2/S2_<id>.npy         int16, shape (T, 10, 128, 128), reflectance x 10000
      ANNOTATIONS/TARGET_<id>.npy shape (3, 128, 128); channel 0 = semantic class
      INSTANCE_ANNOTATIONS/INSTANCES_<id>.npy  parcel id per pixel (0 = no parcel)

Band order in DATA_S2: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12.
Classes: 0 = background, 1..18 = crops, 19 = void (ignored).
"""
import json
import os
from datetime import date

import numpy as np

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
RGB_IDX = [2, 1, 0]            # B4 (red), B3 (green), B2 (blue) -> image R, G, B
NON_RGB_IDX = [3, 4, 5, 6, 7, 8, 9]  # B5, B6, B7, B8, B8A, B11, B12
RED_IDX, NIR_IDX, BLUE_IDX = 2, 6, 0
NUM_CLASSES = 19               # classes 0..18 are evaluated
VOID_LABEL = 19                # ignored in loss and metrics
REF_DATE = date(2018, 9, 1)    # dates are "days since Sept 1, 2018"

CLASS_NAMES = [
    "Background", "Meadow", "Soft winter wheat", "Corn", "Winter barley",
    "Winter rapeseed", "Spring barley", "Sunflower", "Grapevine", "Beet",
    "Winter triticale", "Winter durum wheat", "Fruits, vegetables, flowers",
    "Potatoes", "Leguminous fodder", "Soybeans", "Orchard", "Mixed cereal",
    "Sorghum", "Void label",
]


def _to_days(yyyymmdd):
    s = str(int(yyyymmdd))
    d = date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    return (d - REF_DATE).days


def load_metadata(pastis_dir):
    """Return {patch_id: {"fold": int, "tile": str, "days": np.int32 array (T,)}}.

    Reads metadata.geojson with the json module only (no geopandas needed).
    Dates are sorted by their index key, which is the order of the T axis in DATA_S2.
    """
    with open(os.path.join(pastis_dir, "metadata.geojson")) as f:
        gj = json.load(f)
    meta = {}
    for feat in gj["features"]:
        p = feat["properties"]
        pid = int(p["ID_PATCH"])
        dates = p["dates-S2"]
        if isinstance(dates, str):
            dates = json.loads(dates)
        ordered = [dates[k] for k in sorted(dates.keys(), key=lambda k: int(k))]
        meta[pid] = {
            "fold": int(p["Fold"]),
            "tile": str(p.get("TILE", "")),
            "days": np.array([_to_days(d) for d in ordered], dtype=np.int32),
        }
    return dict(sorted(meta.items()))


def load_band_norm(pastis_dir, folds=(1, 2, 3)):
    """Per-band mean/std (10 values each) averaged over the given folds, as in the official loader."""
    with open(os.path.join(pastis_dir, "NORM_S2_patch.json")) as f:
        nv = json.load(f)
    mean = np.stack([np.array(nv[f"Fold_{k}"]["mean"], dtype=np.float32) for k in folds]).mean(0)
    std = np.stack([np.array(nv[f"Fold_{k}"]["std"], dtype=np.float32) for k in folds]).mean(0)
    return mean, std


def s2_path(pastis_dir, pid):
    return os.path.join(pastis_dir, "DATA_S2", f"S2_{pid}.npy")


def load_s2(pastis_dir, pid, mmap=True):
    """(T, 10, 128, 128) int16 array (memory-mapped by default)."""
    return np.load(s2_path(pastis_dir, pid), mmap_mode="r" if mmap else None)


def load_target(pastis_dir, pid):
    t = np.load(os.path.join(pastis_dir, "ANNOTATIONS", f"TARGET_{pid}.npy"))
    return (t[0] if t.ndim == 3 else t).astype(np.int64)


def load_parcels(pastis_dir, pid):
    """Parcel id per pixel (0 = no parcel). Used for the field-level score and the NDVI probe."""
    p = os.path.join(pastis_dir, "INSTANCE_ANNOTATIONS", f"INSTANCES_{pid}.npy")
    if not os.path.exists(p):
        p = os.path.join(pastis_dir, "ANNOTATIONS", f"ParcelIDs_{pid}.npy")
    return np.load(p).astype(np.int64)


def load_splits(path):
    with open(path) as f:
        s = json.load(f)
    return {k: [int(i) for i in v] for k, v in s.items() if isinstance(v, list)}


def make_splits(meta, fractions=(0.1, 0.5), seed=0):
    """Official folds: train = 1-3, val = 4, test = 5. Label subsets are fixed random subsets of train."""
    train = [p for p, m in meta.items() if m["fold"] in (1, 2, 3)]
    splits = {
        "train": train,
        "val": [p for p, m in meta.items() if m["fold"] == 4],
        "test": [p for p, m in meta.items() if m["fold"] == 5],
    }
    rng = np.random.RandomState(seed)
    perm = rng.permutation(train)
    for frac in fractions:
        n = max(1, int(round(frac * len(train))))
        splits[f"train_{int(round(frac * 100))}"] = sorted(int(x) for x in perm[:n])
    return splits
