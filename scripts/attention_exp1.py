"""Does our L-TAE attend to clear dates, or to certain months? (Experiment 1, as Pair 1's
attention_analysis.py, but for runs of scripts/train_temporal.py.)

For each run (L-TAE trained with real dates on all dates), the best checkpoint is applied to the
test fold with real dates. Attention of each date = mean over heads and the 16x16 tokens, times
the number of dates T (1.0 = uniform attention). A date is cloudy if more than 20% of its pixels
have blue > 2000 (same rule as the cloud-free runs).

    python scripts/attention_exp1.py --pastis $PASTIS --splits $SPLITS --feats-root $FEATS \
        --clouds $FEATS/cloud_fractions.npz --out $RUNS/exp1_ours/attention \
        --runs $RUNS/exp1_ours/dinov3_sat__ltae__real__train__s*
"""
import argparse
import json
import os
import sys
from datetime import timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.clouds import cloud_fractions  # noqa: E402
from src.data import FeatureSeriesDataset, pad_collate  # noqa: E402
from src.pastis import NUM_CLASSES, REF_DATE, load_metadata, load_splits  # noqa: E402
from src.temporal import SegModel  # noqa: E402
from src.utils import autocast, get_device, save_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pastis", required=True)
ap.add_argument("--splits", required=True)
ap.add_argument("--feats-root", required=True)
ap.add_argument("--clouds", required=True)
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cloudy-frac", type=float, default=0.2)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--max-series", type=int, default=0)
a = ap.parse_args()

device = get_device("auto")
meta = load_metadata(a.pastis)
test_ids = load_splits(a.splits)["test"]
if a.max_series:
    test_ids = test_ids[: a.max_series]
clouds = cloud_fractions(a.pastis, meta, a.clouds)
os.makedirs(a.out, exist_ok=True)
MONTHS = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def analyse(att, cloud, days):
    clear = cloud <= a.cloudy_frac
    month = np.array([(REF_DATE + timedelta(days=int(d))).month for d in days])
    by_month = {int(m): float(att[clear & (month == m)].mean()) if (clear & (month == m)).any() else None
                for m in MONTHS}
    return {"n_dates": int(len(att)), "share_cloudy": float((~clear).mean()),
            "attention_clear": float(att[clear].mean()), "attention_cloudy": float(att[~clear].mean()),
            "spearman_attention_cloud": spearman(att, cloud), "attention_by_month_clear_only": by_month}


per_run, pooled = {}, {"att": [], "cloud": [], "days": []}
for run in a.runs:
    name = os.path.basename(os.path.normpath(run))
    res = json.load(open(os.path.join(run, "results.json")))
    args = res["args"]
    assert args["temporal"] == "ltae" and args["dates"] == "real" and args.get("select", "all") == "all", name
    ck = torch.load(os.path.join(run, "best.pt"), map_location="cpu", weights_only=False)
    model = SegModel(ck["in_dim"], temporal="ltae", num_classes=NUM_CLASSES, use_dates=True, return_att=True)
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    ds = FeatureSeriesDataset(os.path.join(a.feats_root, res["feature_name"]), a.pastis, test_ids, meta,
                              date_mode="real")
    att, cl, dy = [], [], []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers, collate_fn=pad_collate):
            with autocast(device):
                _, w = model(b["x"].to(device).float(), b["days"].to(device), b["pad_mask"].to(device))
            w = w.float().mean(dim=(0, 2, 3)).cpu().numpy()            # heads,B,h,w,T -> B,T
            for i in range(w.shape[0]):
                keep = ~b["pad_mask"][i].numpy()
                T = int(keep.sum())
                pid = int(b["pid"][i])
                att.append(w[i][keep] * T)
                cl.append(clouds[pid][:T])
                dy.append(b["days"][i].numpy()[keep])
                assert len(clouds[pid]) == T
    att, cl, dy = map(np.concatenate, (att, cl, dy))
    per_run[name] = analyse(att, cl, dy)
    for k, v in (("att", att), ("cloud", cl), ("days", dy)):
        pooled[k].append(v)
    r = per_run[name]
    print(f"{name}: attention clear {r['attention_clear']:.2f}, cloudy {r['attention_cloudy']:.2f}, "
          f"Spearman {r['spearman_attention_cloud']:.2f}", flush=True)

att, cl, dy = (np.concatenate(pooled[k]) for k in ("att", "cloud", "days"))
summary = analyse(att, cl, dy)
save_json({"runs": per_run, "pooled": summary, "cloudy_frac": a.cloudy_frac,
           "note": "attention x T (1 = uniform), mean over heads and tokens, test fold 5, real dates"},
          os.path.join(a.out, "attention_exp1.json"))
lines = [f"Runs: {len(per_run)} (pooled). Dates: {summary['n_dates']}, cloudy: {100 * summary['share_cloudy']:.1f}%",
         f"relative attention, clear dates:  {summary['attention_clear']:.2f}",
         f"relative attention, cloudy dates: {summary['attention_cloudy']:.2f}",
         f"Spearman(attention, cloud fraction): {summary['spearman_attention_cloud']:.2f}",
         "by month (clear dates only): " + ", ".join(
             f"{m}: {v:.2f}" for m, v in summary["attention_by_month_clear_only"].items() if v is not None)]
open(os.path.join(a.out, "attention_exp1.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(cl, att, s=2, alpha=0.2)
ax[0].set_xlabel("cloud fraction")
ax[0].set_ylabel("attention × T")
ax[0].axhline(1, color="k", lw=0.5)
ax[0].set_title("Attention vs clouds")
bm = summary["attention_by_month_clear_only"]
ax[1].bar(range(12), [bm[m] or 0 for m in MONTHS])
ax[1].set_xticks(range(12))
ax[1].set_xticklabels(["S", "O", "N", "D", "J", "F", "M", "A", "M", "J", "J", "A"])
ax[1].axhline(1, color="k", lw=0.5)
ax[1].set_title("Attention by month, clear dates only")
plt.tight_layout()
plt.savefig(os.path.join(a.out, "fig_attention_exp1.png"), dpi=150)
print("saved", a.out)
