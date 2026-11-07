"""Official pre-trained U-TAE (trained from scratch on all 10 bands) on the PASTIS test fold,
with real dates and with shuffled dates at test time. Nothing is trained.

Weights: Zenodo record 5172293, UTAE_SemanticSeg_weights.zip (md5 91bf47d219927b2a686ec662132f754f).
Its "Fold_1" model was trained on folds 1-3 and validated on fold 4 (official fold_sequence[0]),
i.e. exactly our split, so fold 5 is its test fold.
Code: the official repository github.com/VSainteuf/utae-paps (MIT), given with --utae-code.

Same as the official test: inputs standardised with the NORM_S2_patch.json values of the
test fold, dates = days since 2018-09-01. Differences: batch size 1 (no padding), and the
prediction is the best of classes 0-18 (void can not be predicted, as for our L-TAE models).
The shuffled-date permutation of a series is np.random.RandomState(pid).permutation(T), the
same one our feature models get at test time (FeatureSeriesDataset, seed 0).

This script does not import this repository's `src` package (the official code also has a
package called `src`). Output: <out>/utae_pretrained.npz with per-series confusion matrices.

    python scripts/eval_utae_pretrained.py --utae-code third_party/utae-paps \
        --weights $DATA_DIR/utae_weights --pastis $PASTIS --splits $SPLITS --out $RUNS/reliance
"""
import argparse
import glob
import json
import os
import sys
from datetime import date

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--utae-code", required=True, help="folder of the official utae-paps repository")
ap.add_argument("--weights", required=True, help="unzipped weights folder (contains conf.json and Fold_1/)")
ap.add_argument("--pastis", required=True)
ap.add_argument("--splits", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fold-dir", default="Fold_1", help="Fold_1 = train 1-3, val 4, test 5")
ap.add_argument("--max-series", type=int, default=0)
ap.add_argument("--min-real-miou", type=float, default=50.0,
                help="stop if the real-date mIoU is below this (wrong normalisation or dates)")
a = ap.parse_args()

sys.path.insert(0, os.path.abspath(a.utae_code))
from src.backbones.utae import UTAE  # noqa: E402  (official code)

NUM_CLASSES, VOID = 19, 19
REF = date(2018, 9, 1)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpts = sorted(glob.glob(os.path.join(a.weights, "**", a.fold_dir, "model.pth.tar"), recursive=True))
assert len(ckpts) == 1, f"expected one {a.fold_dir}/model.pth.tar under {a.weights}, found {ckpts}"
fold_path = os.path.dirname(ckpts[0])
conf_path = next(p for p in (os.path.join(fold_path, "conf.json"), os.path.join(os.path.dirname(fold_path), "conf.json"))
                 if os.path.exists(p))
cfg = json.load(open(conf_path))
print("weights:", ckpts[0], "| config:", conf_path)
lst = lambda v: json.loads(v) if isinstance(v, str) else v  # noqa: E731
assert cfg.get("model", "utae") == "utae", cfg.get("model")
model = UTAE(input_dim=10, encoder_widths=lst(cfg["encoder_widths"]), decoder_widths=lst(cfg["decoder_widths"]),
             out_conv=lst(cfg["out_conv"]), str_conv_k=cfg["str_conv_k"], str_conv_s=cfg["str_conv_s"],
             str_conv_p=cfg["str_conv_p"], agg_mode=cfg["agg_mode"], encoder_norm=cfg["encoder_norm"],
             n_head=cfg["n_head"], d_model=cfg["d_model"], d_k=cfg["d_k"], encoder=False,
             return_maps=False, pad_value=cfg["pad_value"], padding_mode=cfg["padding_mode"])
sd = torch.load(ckpts[0], map_location="cpu", weights_only=False)
model.load_state_dict(sd["state_dict"])
model.eval().to(dev)

with open(os.path.join(a.pastis, "metadata.geojson")) as f:
    feats = json.load(f)["features"]
days_of = {}
for ft in feats:
    p = ft["properties"]
    ds = p["dates-S2"]
    ds = json.loads(ds) if isinstance(ds, str) else ds
    ordered = [str(ds[k]) for k in sorted(ds, key=lambda k: int(k))]
    days_of[int(p["ID_PATCH"])] = np.array(
        [(date(int(s[:4]), int(s[4:6]), int(s[6:8])) - REF).days for s in ordered], dtype=np.float32)
norm = json.load(open(os.path.join(a.pastis, "NORM_S2_patch.json")))["Fold_5"]
mean = torch.tensor(norm["mean"], dtype=torch.float32)[None, :, None, None]
std = torch.tensor(norm["std"], dtype=torch.float32)[None, :, None, None]
test_ids = [int(i) for i in json.load(open(a.splits))["test"]]
if a.max_series:
    test_ids = test_ids[: a.max_series]


def conf(pred, y):
    m = y != VOID
    return np.bincount(y[m] * NUM_CLASSES + pred[m], minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def miou(cm):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - tp
    return 100 * np.nanmean(np.where(union > 0, tp / np.maximum(union, 1), np.nan))


cms = {"real": [], "shuffled": []}
with torch.no_grad():
    for k, pid in enumerate(test_ids):
        x = torch.from_numpy(np.load(os.path.join(a.pastis, "DATA_S2", f"S2_{pid}.npy")).astype(np.float32))
        x = ((x - mean) / std)[None].to(dev)                                  # 1, T, 10, 128, 128
        y = np.load(os.path.join(a.pastis, "ANNOTATIONS", f"TARGET_{pid}.npy"))[0].astype(np.int64)
        d = days_of[pid]
        assert len(d) == x.shape[1], f"{pid}: dates and images disagree"
        perm = np.random.RandomState(pid).permutation(len(d))                # same as our models
        for mode, dd in (("real", d), ("shuffled", d[perm])):
            out = model(x, batch_positions=torch.from_numpy(dd)[None].to(dev))
            pred = out[0, :NUM_CLASSES].argmax(0).cpu().numpy()
            cms[mode].append(conf(pred, y))
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(test_ids)}: real {miou(np.sum(cms['real'], 0)):.2f}, "
                  f"shuffled {miou(np.sum(cms['shuffled'], 0)):.2f}", flush=True)

cr, cs = np.stack(cms["real"]), np.stack(cms["shuffled"])
m_real, m_shuf = miou(cr.sum(0)), miou(cs.sum(0))
print(f"U-TAE pretrained ({a.fold_dir}) on fold 5: real {m_real:.2f}, shuffled at test {m_shuf:.2f}, "
      f"drop {m_real - m_shuf:.2f}")
os.makedirs(a.out, exist_ok=True)
np.savez_compressed(os.path.join(a.out, "utae_pretrained.npz"), cms_real=cr, cms_shuffled=cs,
                    pids=np.array(test_ids))
if not a.max_series and m_real < a.min_real_miou:
    sys.exit(f"FAIL real-date mIoU {m_real:.2f} < {a.min_real_miou}: check normalisation / date order")
print("saved", os.path.join(a.out, "utae_pretrained.npz"))
