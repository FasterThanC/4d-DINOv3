"""PyTorch datasets.

FeatureSeriesDataset  - saved token features of a whole time series + crop labels
                        (for the temporal model / crop map).
ImageDistillDataset   - single Sentinel-2 images + the saved teacher tokens of the same image
                        (for training the student).

Saved feature format (shared with Pair 1):
    <feat_dir>/<patch_id>.npy       float16, shape (T, D, 16, 16), T in the same order as DATA_S2
    <feat_dir>/<patch_id>_days.npy  int32, shape (T,), days since 2018-09-01
    <feat_dir>/manifest.json        model, input size, scaling, mean/std
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .pastis import NON_RGB_IDX, load_parcels, load_s2, load_target


def feat_path(feat_dir, pid):
    return os.path.join(feat_dir, f"{pid}.npy")


class FeatureSeriesDataset(Dataset):
    """Returns dict(x=(T, D, h, w) float16, days=(T,), y=(128,128), parcels=(128,128), pid)."""

    def __init__(self, feat_dir, pastis_dir, ids, meta, train=False, max_dates=32,
                 date_mode="real", cache_ram=False, with_parcels=False, seed=0):
        assert date_mode in ("real", "shuffled", "none")
        self.feat_dir, self.pastis_dir = feat_dir, pastis_dir
        self.ids, self.meta = list(ids), meta
        self.train, self.max_dates, self.date_mode = train, max_dates, date_mode
        self.with_parcels, self.seed = with_parcels, seed
        self.cache = {}
        self.targets = {pid: load_target(pastis_dir, pid) for pid in self.ids}
        if cache_ram:
            for pid in self.ids:
                self.cache[pid] = np.load(feat_path(feat_dir, pid))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        pid = self.ids[i]
        feats = self.cache.get(pid)
        if feats is None:
            feats = np.load(feat_path(self.feat_dir, pid), mmap_mode="r")
        days = self.meta[pid]["days"].astype(np.float32)
        T = feats.shape[0]
        assert T == len(days), f"patch {pid}: {T} feature dates vs {len(days)} metadata dates"
        if self.train and self.max_dates and T > self.max_dates:
            keep = torch.randperm(T)[: self.max_dates].sort().values.numpy()
        else:
            keep = np.arange(T)
        x = torch.from_numpy(np.ascontiguousarray(feats[keep]))
        days = days[keep]
        if self.date_mode == "shuffled":
            # The model sees the same images, but every image gets another image's date.
            if self.train:
                perm = torch.randperm(len(days)).numpy()
            else:                                    # fixed per patch, so tests are repeatable
                perm = np.random.RandomState(self.seed * 100003 + pid).permutation(len(days))
            days = days[perm]
        elif self.date_mode == "none":
            days = np.zeros_like(days)
        out = {"x": x, "days": torch.from_numpy(days), "y": torch.from_numpy(self.targets[pid]),
               "pid": pid}
        if self.with_parcels:
            out["parcels"] = torch.from_numpy(load_parcels(self.pastis_dir, pid))
        return out


def pad_collate(batch):
    T = max(b["x"].shape[0] for b in batch)
    B = len(batch)
    D, h, w = batch[0]["x"].shape[1:]
    x = torch.zeros(B, T, D, h, w, dtype=batch[0]["x"].dtype)
    days = torch.zeros(B, T)
    pad = torch.ones(B, T, dtype=torch.bool)
    for i, b in enumerate(batch):
        t = b["x"].shape[0]
        x[i, :t], days[i, :t], pad[i, :t] = b["x"], b["days"], False
    out = {"x": x, "days": days, "pad_mask": pad, "y": torch.stack([b["y"] for b in batch]),
           "pid": torch.tensor([b["pid"] for b in batch])}
    if "parcels" in batch[0]:
        out["parcels"] = torch.stack([b["parcels"] for b in batch])
    return out


class ImageDistillDataset(Dataset):
    """One item = one image (patch, date): standardised 10 bands, teacher tokens, rebuild target."""

    def __init__(self, pastis_dir, teacher_dir, ids, meta, band_mean, band_std, max_items=None, seed=0):
        self.pastis_dir, self.teacher_dir = pastis_dir, teacher_dir
        self.items = [(pid, t) for pid in ids for t in range(len(meta[pid]["days"]))]
        if max_items and len(self.items) > max_items:
            rng = np.random.RandomState(seed)
            self.items = [self.items[j] for j in sorted(rng.choice(len(self.items), max_items, replace=False))]
        self.mean = torch.tensor(band_mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(band_std, dtype=torch.float32).view(-1, 1, 1)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        pid, t = self.items[i]
        img = torch.from_numpy(np.asarray(load_s2(self.pastis_dir, pid)[t], dtype=np.float32))
        x = (img - self.mean) / self.std                                    # 10, 128, 128
        z = np.asarray(np.load(feat_path(self.teacher_dir, pid), mmap_mode="r")[t])   # D, h, w
        z = torch.from_numpy(z.astype(np.float32)).flatten(1).T             # N, D
        return {"x": x, "z": z, "rebuild": x[NON_RGB_IDX].clone()}
