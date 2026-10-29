"""Save the trained student's tokens for every image (same format as the DINOv3 features).

    python scripts/extract_student.py --pastis $PASTIS --ckpt $RUNS/students/student2_rebuild/best.pt \
        --out $FEATS/student2_rebuild
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pastis import load_band_norm, load_metadata, load_s2  # noqa: E402
from src.student import student_from_checkpoint  # noqa: E402
from src.utils import autocast, get_device, load_json, save_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pastis", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--projected", action="store_true",
                help="save the 1024-d projected tokens instead of the 384-d student tokens")
ap.add_argument("--device", default="auto")
ap.add_argument("--ids-from", default="", help="splits json: only extract the series listed in it")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

device = get_device(a.device)
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
init = ck.get("init", "")
model = student_from_checkpoint(ck)
model.eval().to(device)

meta = load_metadata(a.pastis)
mean, std = load_band_norm(a.pastis, folds=(1, 2, 3))
mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)
os.makedirs(a.out, exist_ok=True)
save_json({"model": "student", "ckpt": os.path.abspath(a.ckpt), "init": init,
           "projected": a.projected, "dtype_saved": "float16", "shape": "(T, D, h, w)"},
          os.path.join(a.out, "manifest.json"))

ids = list(meta.keys())[: a.limit or None]
if a.ids_from:                      # only the series listed in a splits file (pilot runs)
    wanted = {int(i) for v in load_json(a.ids_from).values() if isinstance(v, list) for i in v}
    ids = [p for p in ids if p in wanted]
t0, n = time.time(), 0
for k, pid in enumerate(ids):
    dst = os.path.join(a.out, f"{pid}.npy")
    if os.path.exists(dst):
        continue
    s2 = load_s2(a.pastis, pid)
    chunks = []
    for s in range(0, s2.shape[0], a.batch):
        x = torch.from_numpy(np.asarray(s2[s:s + a.batch], dtype=np.float32)).to(device)
        x = (x - mean_t) / std_t
        with torch.inference_mode(), autocast(device):
            tok, proj, _ = model(x)
            if a.projected:
                B, D, h, w = tok.shape
                tok = proj.transpose(1, 2).reshape(B, -1, h, w)
        chunks.append(tok.float().cpu().numpy().astype(np.float16))
    feats = np.concatenate(chunks)
    assert feats.shape[0] == len(meta[pid]["days"]), f"patch {pid}: T does not match metadata"
    assert np.isfinite(feats).all(), f"patch {pid}: NaN/inf in student tokens"
    np.save(dst[:-4] + ".tmp.npy", feats)
    os.replace(dst[:-4] + ".tmp.npy", dst)
    np.save(os.path.join(a.out, f"{pid}_days.npy"), meta[pid]["days"])
    n += feats.shape[0]
    if (k + 1) % 50 == 0 or k + 1 == len(ids):
        print(f"{k + 1}/{len(ids)} series, {n / max(time.time() - t0, 1e-6):.1f} img/s", flush=True)
print("done:", a.out)
