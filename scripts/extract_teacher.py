"""Save frozen DINOv3 patch tokens for every image of every PASTIS series.

Output per series: <out>/<id>.npy (T, D, 16, 16) float16 and <out>/<id>_days.npy.
Already finished series are skipped, so the job can be stopped and restarted.

Quick start (first ~1,000 images so Pair 1 can start):
    python scripts/extract_teacher.py --pastis $PASTIS --out $FEATS/dinov3_sat --limit 25
Full run:
    python scripts/extract_teacher.py --pastis $PASTIS --out $FEATS/dinov3_sat
Stand-in while DINOv3 access is pending (open weights, same 16 x 16 x 1024 output):
    python scripts/extract_teacher.py --model facebook/dinov2-large --size 224 --out $FEATS/dinov2_large ...
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dino import (compute_rgb_percentiles, default_mean_std, load_backbone,  # noqa: E402
                      patch_tokens, prepare_rgb)
from src.pastis import load_metadata, load_s2  # noqa: E402
from src.utils import get_device, load_json, save_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pastis", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-sat493m")
ap.add_argument("--size", type=int, default=256, help="input size after upsampling (256 = x2)")
ap.add_argument("--pct", default=None, help="RGB percentile json; computed from folds 1-3 if missing")
ap.add_argument("--batch", type=int, default=64, help="images per forward pass")
ap.add_argument("--folds", default="1,2,3,4,5")
ap.add_argument("--limit", type=int, default=0, help="only the first N series (quick start)")
ap.add_argument("--device", default="auto")
ap.add_argument("--ids-from", default="", help="splits json: only extract the series listed in it")
ap.add_argument("--fp32", action="store_true", help="disable bf16 inference")
ap.add_argument("--norm", default="auto", choices=["auto", "sat", "web"],
                help="mean/std: sat (SAT-493M models) or web; auto picks from the model name")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
device = get_device(a.device)
meta = load_metadata(a.pastis)
folds = {int(f) for f in a.folds.split(",")}
ids = [p for p, m in meta.items() if m["fold"] in folds]
if a.limit:
    ids = ids[: a.limit]
if a.ids_from:                      # only the series listed in a splits file (pilot runs)
    wanted = {int(i) for v in load_json(a.ids_from).values() if isinstance(v, list) for i in v}
    ids = [p for p in ids if p in wanted]

pct_path = a.pct or os.path.join(a.out, "rgb_percentiles.json")
if os.path.exists(pct_path):
    pct = load_json(pct_path)
else:
    train_ids = [p for p, m in meta.items() if m["fold"] in (1, 2, 3)]
    rng = np.random.RandomState(0)
    sample = rng.choice(train_ids, size=min(200, len(train_ids)), replace=False)
    pct = compute_rgb_percentiles([load_s2(a.pastis, int(p)) for p in sample])
    save_json(pct, pct_path)
    print("RGB percentiles (from folds 1-3):", pct)

mean, std = default_mean_std(a.model, a.norm)
dtype = torch.float32 if (a.fp32 or device.type != "cuda") else torch.bfloat16
model = load_backbone(a.model, device, dtype=dtype)
save_json({"model": a.model, "input_size": a.size, "rgb_bands": ["B4", "B3", "B2"],
           "percentiles": pct, "mean": mean, "std": std, "dtype_saved": "float16",
           "shape": "(T, D, h, w)", "dates": "<id>_days.npy, days since 2018-09-01"},
          os.path.join(a.out, "manifest.json"))

t0, n_img = time.time(), 0
for k, pid in enumerate(ids):
    dst = os.path.join(a.out, f"{pid}.npy")
    if os.path.exists(dst):
        continue
    s2 = load_s2(a.pastis, pid)
    chunks = []
    for s in range(0, s2.shape[0], a.batch):
        x = torch.from_numpy(np.asarray(s2[s:s + a.batch], dtype=np.float32)).to(device)
        x = prepare_rgb(x, pct, mean, std, size=a.size).to(dtype)
        with torch.inference_mode():
            chunks.append(patch_tokens(model, x).float().cpu().numpy().astype(np.float16))
    feats = np.concatenate(chunks)
    assert np.isfinite(feats).all(), f"patch {pid}: NaN/inf in tokens (overflow in float16?)"
    assert feats.shape[0] == len(meta[pid]["days"]), \
        f"patch {pid}: {feats.shape[0]} images but {len(meta[pid]['days'])} dates in metadata"
    np.save(os.path.join(a.out, f"{pid}_days.npy"), meta[pid]["days"])
    tmp = dst[:-4] + ".tmp.npy"
    np.save(tmp, feats)
    os.replace(tmp, dst)          # the feature file appears only when complete
    n_img += feats.shape[0]
    if (k + 1) % 20 == 0 or k + 1 == len(ids):
        rate = n_img / max(time.time() - t0, 1e-6)
        print(f"{k + 1}/{len(ids)} series, {n_img} images, {rate:.1f} img/s", flush=True)
print("done:", a.out)
