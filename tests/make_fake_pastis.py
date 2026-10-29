"""Make a tiny fake PASTIS folder + tiny random DINOv3 models, to test the code without
downloading anything (runs on a laptop CPU in a few minutes).

    python tests/make_fake_pastis.py --out /tmp/fake
"""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--patches", type=int, default=30)
a = ap.parse_args()
rng = np.random.RandomState(0)

root = os.path.join(a.out, "PASTIS")
for d in ("DATA_S2", "ANNOTATIONS", "INSTANCE_ANNOTATIONS"):
    os.makedirs(os.path.join(root, d), exist_ok=True)

features, sums, sqs, cnt = [], np.zeros(10), np.zeros(10), 0
peak = {c: rng.uniform(60, 350) for c in range(20)}       # day of greenest growth per crop
peak[0] = 5000                                             # background: never green
for k in range(a.patches):
    pid = 10000 + k
    T = rng.randint(8, 13)
    days = np.sort(rng.choice(np.arange(10, 420), T, replace=False))
    dates = {}
    for i, d in enumerate(days):
        dt = np.datetime64("2018-09-01") + np.timedelta64(int(d), "D")
        dates[str(i)] = int(str(dt).replace("-", ""))
    sem = np.zeros((128, 128), dtype=np.int64)
    par = np.zeros((128, 128), dtype=np.int64)
    pidn = 1
    for _ in range(rng.randint(4, 9)):
        y0, x0 = rng.randint(0, 100, 2)
        hh, ww = rng.randint(24, 56, 2)
        sem[y0:y0 + hh, x0:x0 + ww] = rng.randint(1, 7)      # 6 crop classes, big fields
        par[y0:y0 + hh, x0:x0 + ww] = pidn
        pidn += 1
    sem[rng.rand(128, 128) < 0.01] = 19
    img = np.zeros((T, 10, 128, 128), dtype=np.float32)
    for t, d in enumerate(days):
        green = np.zeros((128, 128))
        for c in np.unique(sem):
            green[sem == c] = np.exp(-((d - peak[c]) / 60.0) ** 2)
        base = 800 + 200 * rng.rand(10)
        for b in range(10):
            img[t, b] = base[b] + (2500 * green if b >= 6 else -300 * green) + 50 * rng.randn(128, 128)
    img = np.clip(img, 0, 10000).astype(np.int16)
    np.save(os.path.join(root, "DATA_S2", f"S2_{pid}.npy"), img)
    np.save(os.path.join(root, "ANNOTATIONS", f"TARGET_{pid}.npy"), np.stack([sem, sem, sem]))
    np.save(os.path.join(root, "INSTANCE_ANNOTATIONS", f"INSTANCES_{pid}.npy"), par)
    fold = k % 5 + 1
    features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]},
                     "properties": {"ID_PATCH": pid, "Fold": fold, "TILE": "T31TFM",
                                    "dates-S2": json.dumps(dates)}})
    f = img.astype(np.float64)
    sums += f.mean(axis=(0, 2, 3))
    sqs += f.std(axis=(0, 2, 3))
    cnt += 1

with open(os.path.join(root, "metadata.geojson"), "w") as fh:
    json.dump({"type": "FeatureCollection", "features": features}, fh)
norm = {f"Fold_{k}": {"mean": list(sums / cnt), "std": list(sqs / cnt)} for k in range(1, 6)}
with open(os.path.join(root, "NORM_S2_patch.json"), "w") as fh:
    json.dump(norm, fh)

from transformers import DINOv3ViTConfig, DINOv3ViTModel  # noqa: E402

small = dict(num_hidden_layers=2, intermediate_size=128, patch_size=16, num_register_tokens=4)
DINOv3ViTModel(DINOv3ViTConfig(hidden_size=64, num_attention_heads=4, **small)).save_pretrained(
    os.path.join(a.out, "tiny_teacher"))
DINOv3ViTModel(DINOv3ViTConfig(hidden_size=32, num_attention_heads=4, **small)).save_pretrained(
    os.path.join(a.out, "tiny_student_init"))
print("fake data and tiny models in", a.out)
