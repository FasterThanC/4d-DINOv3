"""PASTIS + saved DINOv3 features.

Expects in feat_dir, per series id:
  <id>.npy        float16 [T, D, h, w]   DINOv3 patch features
  <id>_days.npy   int     [T]            days since 2018-09-01
  <id>_cloud.npy  float32 [T]            cloud fraction (from compute_clouds.py)
and in the PASTIS root:
  metadata.geojson            (ID_PATCH, Fold)
  ANNOTATIONS/TARGET_<id>.npy [3,128,128], channel 0 = semantic class (0..19, 19 = void)
"""
import datetime as dt
import json, os
import numpy as np
import torch
from torch.utils.data import Dataset

REF_DATE = dt.date(2018, 9, 1)
VOID = 19
NUM_CLASSES = 20          # 0 background, 1..18 crops, 19 void (ignored)


def read_meta(pastis_root):
    meta = json.load(open(os.path.join(pastis_root, "metadata.geojson")))
    return [(f["properties"]["ID_PATCH"], int(f["properties"]["Fold"])) for f in meta["features"]]


def select_dates(days, cloud, args, rng):
    """Return (indices into the series, positions, dates) for the dates the model will see.

    args.dates:      'all' | 'clear'
    args.agg:        'single' picks one date (least cloudy in args.month), else keeps a sequence
    args.pe:         controls what 'positions' means (renumbered after cloud removal)
    """
    T = len(days)
    idx = np.arange(T)

    if args.agg == "single":
        months = np.array([(REF_DATE + dt.timedelta(int(d))).month for d in days])
        cand = idx[months == args.month]
        if len(cand) == 0:  # no acquisition that month: take the nearest date to the 15th
            target = _days_of_month_mid(days, args.month)
            cand = idx[[np.argmin(np.abs(days - target))]]
        idx = cand[[np.argmin(cloud[cand])]]

    elif args.dates == "clear":
        clear = idx[cloud <= args.cloudy_frac]
        if len(clear) < args.min_dates:  # keep at least min_dates, the least cloudy ones
            clear = np.sort(np.argsort(cloud, kind="stable")[: args.min_dates])
        idx = clear

    sel_days = days[idx].astype(np.float32)
    # 'position' = pure order of the dates the model actually sees, renumbered 0,1,2,...
    # scaled by pos_scale so the sinusoids cover a similar range as the date version
    positions = np.arange(len(idx), dtype=np.float32) * args.pos_scale

    if args.pe == "shuffled":  # wrong dates, fresh permutation every time a sample is loaded
        sel_days = rng.permutation(sel_days)

    return idx, positions, sel_days


def _days_of_month_mid(days, month):
    year = 2018 if month >= 9 else 2019
    return (dt.date(year, month, 15) - REF_DATE).days


class PastisFeatures(Dataset):
    def __init__(self, pastis_root, feat_dir, folds, args, label_frac=1.0, label_seed=0):
        self.pastis_root, self.feat_dir, self.args = pastis_root, feat_dir, args
        self.ids = [pid for pid, f in read_meta(pastis_root) if f in folds]
        if label_frac < 1.0:  # fixed subset, same for every run with the same label_seed
            r = np.random.default_rng(label_seed)
            keep = r.choice(len(self.ids), int(round(label_frac * len(self.ids))), replace=False)
            self.ids = [self.ids[i] for i in sorted(keep)]
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        pid = self.ids[i]
        p = lambda s: os.path.join(self.feat_dir, f"{pid}{s}")
        days = np.load(p("_days.npy")).astype(np.int64)
        cloud = np.load(p("_cloud.npy")).astype(np.float32)
        idx, pos, sel_days = select_dates(days, cloud, self.args, self.rng)

        feats = np.load(p(".npy"), mmap_mode="r")
        x = torch.from_numpy(np.ascontiguousarray(feats[idx]))  # idx is sorted in time
        y = np.load(os.path.join(self.pastis_root, "ANNOTATIONS", f"TARGET_{pid}.npy"))[0]
        return {
            "x": x,                                        # [T', D, h, w] float16
            "pos": torch.from_numpy(pos),                  # [T']
            "days": torch.from_numpy(sel_days),            # [T']
            "cloud": torch.from_numpy(cloud[idx]),         # [T'] for the attention analysis
            "y": torch.from_numpy(y.astype(np.int64)),     # [128,128]
            "id": pid,
        }


def collate(batch):
    """Pad to the longest sequence in the batch; mask=True marks padding."""
    Tm = max(b["x"].shape[0] for b in batch)
    B = len(batch)
    D, h, w = batch[0]["x"].shape[1:]
    x = torch.zeros(B, Tm, D, h, w, dtype=batch[0]["x"].dtype)
    pos = torch.zeros(B, Tm); days = torch.zeros(B, Tm); cloud = torch.zeros(B, Tm)
    pad = torch.ones(B, Tm, dtype=torch.bool)
    for i, b in enumerate(batch):
        T = b["x"].shape[0]
        x[i, :T] = b["x"]; pos[i, :T] = b["pos"]; days[i, :T] = b["days"]
        cloud[i, :T] = b["cloud"]; pad[i, :T] = False
    return {"x": x, "pos": pos, "days": days, "cloud": cloud, "pad": pad,
            "y": torch.stack([b["y"] for b in batch]), "id": [b["id"] for b in batch]}
