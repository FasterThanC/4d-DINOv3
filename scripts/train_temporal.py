"""Train the temporal model + decoder on saved features and test on fold 5.

Works for any feature folder (DINOv3, student 1, student 2, ...).

    python scripts/train_temporal.py --pastis $PASTIS --feats $FEATS/student2_rebuild \
        --splits splits.json --out $RUNS/temporal --train-split train_10 --dates shuffled --seed 0

--train-split  train (100%), train_50 or train_10 (fixed subsets from splits.json)
--dates        real | shuffled (each image gets another image's date) | none (no dates at all)
               | position (only the order of the images, k * --pos-scale) | both (order and date)
--temporal     ltae | mean

All runs use the same number of optimisation steps (--steps), so 10% and 100% get the same
compute. The checkpoint with the best fold-4 mIoU is tested on fold 5.
Outputs in <out>/<run name>/: results.json, log.csv, test_confusions.npz, best.pt
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import FeatureSeriesDataset, pad_collate  # noqa: E402
from src.metrics import confusion, field_hits, scores  # noqa: E402
from src.pastis import CLASS_NAMES, NUM_CLASSES, VOID_LABEL, load_metadata, load_splits  # noqa: E402
from src.temporal import SegModel  # noqa: E402
from src.utils import CSVLogger, Timer, autocast, get_device, git_commit, save_json, set_seed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pastis", required=True)
ap.add_argument("--feats", required=True)
ap.add_argument("--splits", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--name", default="", help="run name (default: built from the settings)")
ap.add_argument("--train-split", default="train")
ap.add_argument("--dates", default="real", choices=["real", "shuffled", "none", "position", "both"])
ap.add_argument("--temporal", default="ltae", choices=["ltae", "mean"])
ap.add_argument("--select", default="all", choices=["all", "cloudfree", "single", "calmonth"],
                help="dates used per series: all, cloud-free only, or one date in --month "
                     "(single: 0 = Sept 2018 ... 14 = Nov 2019; calmonth: calendar month 1-12)")
ap.add_argument("--month", type=int, default=None, help="for --select single / calmonth")
ap.add_argument("--max-cloud", type=float, default=0.1,
                help="a date is cloud-free if at most this share of pixels has blue > 0.2")
ap.add_argument("--min-dates", type=int, default=1, help="cloudfree: keep at least this many (least cloudy) dates")
ap.add_argument("--pos-scale", type=float, default=10.0, help="position/both: image k is encoded as k * pos_scale")
ap.add_argument("--clouds", default="", help="cloud-share file (default: <feats>/../cloud_fractions.npz)")
ap.add_argument("--max-dates", type=int, default=32, help="random dates per series in training")
ap.add_argument("--steps", type=int, default=5000)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--eval-every", type=int, default=500)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--cache-ram", action="store_true", help="load all training features into RAM")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="auto")
a = ap.parse_args()

feat_name = os.path.basename(os.path.normpath(a.feats))
assert a.select not in ("single", "calmonth") or a.month is not None, "--select single/calmonth needs --month"
assert a.temporal == "ltae" or a.dates == "none", "the mean baseline uses no dates: give --dates none"
tname = a.temporal + {"all": "", "cloudfree": "-cloudfree", "single": f"-month{a.month or 0:02d}",
                      "calmonth": f"-cal{a.month or 0:02d}"}[a.select]
name = a.name or f"{feat_name}__{tname}__{a.dates}__{a.train_split}__s{a.seed}"
run_dir = os.path.join(a.out, name)
os.makedirs(run_dir, exist_ok=True)
if os.path.exists(os.path.join(run_dir, "results.json")):
    print("already done:", run_dir)
    sys.exit(0)

set_seed(a.seed)
device = get_device(a.device)
meta = load_metadata(a.pastis)
splits = load_splits(a.splits)

clouds = None
if a.select != "all":
    from src.clouds import cloud_fractions  # noqa: E402
    clouds = cloud_fractions(a.pastis, meta, a.clouds or os.path.join(os.path.dirname(
        os.path.normpath(a.feats)), "cloud_fractions.npz"))
sel = dict(select=a.select, clouds=clouds, month=a.month, max_cloud=a.max_cloud, min_dates=a.min_dates,
           pos_scale=a.pos_scale)
train_ds = FeatureSeriesDataset(a.feats, a.pastis, splits[a.train_split], meta, train=True,
                                max_dates=a.max_dates, date_mode=a.dates, cache_ram=a.cache_ram,
                                seed=a.seed, **sel)
val_ds = FeatureSeriesDataset(a.feats, a.pastis, splits["val"], meta, date_mode=a.dates, seed=a.seed, **sel)
test_ds = FeatureSeriesDataset(a.feats, a.pastis, splits["test"], meta, date_mode=a.dates,
                               with_parcels=True, seed=a.seed, **sel)
in_dim = train_ds[0]["x"].shape[1]
kw = dict(num_workers=a.workers, collate_fn=pad_collate, pin_memory=device.type == "cuda")
train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, drop_last=len(train_ds) > a.batch,
                      persistent_workers=a.workers > 0, **kw)
eval_kw = dict(batch_size=max(1, a.batch // 2), shuffle=False, **kw)

model = SegModel(in_dim, temporal=a.temporal, num_classes=NUM_CLASSES,
                 use_dates=a.dates != "none", time_enc="both" if a.dates == "both" else "date").to(device)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
warm = max(1, int(0.05 * a.steps))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(a.steps - warm, 1))))
print(f"{name}: train {len(train_ds)} series, feature dim {in_dim}, "
      f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M trainable parameters")


def run_eval(ds, per_patch=False, fields=False):
    model.eval()
    cm_total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    per, pids, hits, nf = [], [], 0, 0
    with torch.no_grad():
        for b in DataLoader(ds, **eval_kw):
            x = b["x"].to(device).float()
            with autocast(device):
                logits = model(x, b["days"].to(device), b["pad_mask"].to(device))
            pred = logits.argmax(1).cpu().numpy()
            y = b["y"].numpy()
            for i in range(len(pred)):
                cm = confusion(pred[i], y[i])
                cm_total += cm
                if per_patch:
                    per.append(cm)
                    pids.append(int(b["pid"][i]))
                if fields:
                    c, n = field_hits(pred[i], y[i], b["parcels"][i].numpy())
                    hits, nf = hits + c, nf + n
    model.train()
    out = scores(cm_total)
    if fields:
        out["field_acc"] = 100.0 * hits / max(nf, 1)
        out["n_fields"] = nf
    if per_patch:
        out["_per_patch"] = (np.stack(per), np.array(pids))
    return out


log = CSVLogger(os.path.join(run_dir, "log.csv"))
timer = Timer()
best, step = -1.0, 0
run_loss, run_n = 0.0, 0                     # mean training loss since the last evaluation
model.train()
while step < a.steps:
    for b in train_dl:
        x = b["x"].to(device, non_blocking=True).float()
        y = b["y"].to(device, non_blocking=True)
        with autocast(device):
            logits = model(x, b["days"].to(device), b["pad_mask"].to(device))
        loss = F.cross_entropy(logits.float(), y, ignore_index=VOID_LABEL)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        run_loss, run_n = run_loss + loss.item(), run_n + 1
        if step % 50 == 0:
            print(f"step {step}/{a.steps} loss {loss.item():.4f}", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            v = run_eval(val_ds)
            log.log({"step": step, "loss": round(run_loss / max(run_n, 1), 5), "val_miou": round(v["miou"], 3),
                     "val_oa": round(v["oa"], 3), "hours": round(timer.hours(), 3)})
            run_loss, run_n = 0.0, 0
            print(f"step {step}: val mIoU {v['miou']:.2f}  OA {v['oa']:.2f}", flush=True)
            if v["miou"] > best:
                best = v["miou"]
                torch.save({"model": model.state_dict(), "step": step, "in_dim": in_dim,
                            "args": vars(a)}, os.path.join(run_dir, "best.pt"))
        if step >= a.steps:
            break

model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device)["model"])
t = run_eval(test_ds, per_patch=True, fields=True)
cms, pids = t.pop("_per_patch")
np.savez_compressed(os.path.join(run_dir, "test_confusions.npz"), cms=cms, pids=pids)
results = {
    "run": name, "features": os.path.abspath(a.feats), "feature_name": feat_name,
    "train_split": a.train_split, "dates": a.dates, "temporal": tname, "select": a.select,
    "month": a.month, "max_cloud": a.max_cloud, "min_dates": a.min_dates, "seed": a.seed,
    "val_miou_best": best, "test_miou": t["miou"], "test_oa": t["oa"],
    "test_field_acc": t["field_acc"], "test_n_fields": t["n_fields"],
    "test_iou_per_class": dict(zip(CLASS_NAMES[:NUM_CLASSES], t["iou_per_class"])),
    "gpu_hours": timer.hours() if device.type == "cuda" else 0.0, "hours": timer.hours(),
    "commit": git_commit(), "args": vars(a),
}
save_json(results, os.path.join(run_dir, "results.json"))
print(f"TEST {name}: mIoU {t['miou']:.2f}  OA {t['oa']:.2f}  field acc {t['field_acc']:.2f}")
