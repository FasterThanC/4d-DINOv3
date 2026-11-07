"""Predicted crop maps from trained runs (no retraining needed).

1) Save the predicted maps of one run for the test fold (fold 5):
     python predict.py save runs/ltae_all_date_s0 --pastis /data/PASTIS --feat_dir /data/feats
   -> runs/ltae_all_date_s0/preds/<id>.npy   uint8 [128,128], class per pixel

2) Picture comparing several runs on a few series (uses the saved maps from step 1):
     python predict.py compare runs/single_m06_s0 runs/mean_all_s0 runs/ltae_all_date_s0 \
         --ids 10000 10001 --pastis /data/PASTIS --out compare.png
"""
import argparse, json, os
from argparse import Namespace
import numpy as np


# PASTIS class names (index = label value)
CLASSES = ["Background", "Meadow", "Soft winter wheat", "Corn", "Winter barley", "Winter rapeseed",
           "Spring barley", "Sunflower", "Grapevine", "Beet", "Winter triticale", "Winter durum wheat",
           "Fruits, vegetables, flowers", "Potatoes", "Leguminous fodder", "Soybeans", "Orchard",
           "Mixed cereal", "Sorghum", "Void"]


def save(a):
    import torch
    from torch.utils.data import DataLoader
    from data import PastisFeatures, collate
    from models import TemporalSegmenter

    r = json.load(open(os.path.join(a.run, "result.json")))
    args = Namespace(**r["args"])
    args.pastis = a.pastis or args.pastis
    args.feat_dir = a.feat_dir or args.feat_dir
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TemporalSegmenter(agg=args.agg, pe=args.pe).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.run, "best.pt"), map_location=dev))
    model.eval()

    ds = PastisFeatures(args.pastis, args.feat_dir, a.folds, args)
    dl = DataLoader(ds, batch_size=16, num_workers=a.workers, collate_fn=collate)
    out_dir = os.path.join(a.run, "preds"); os.makedirs(out_dir, exist_ok=True)
    n = 0
    with torch.no_grad():
        for b in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits, _ = model(b["x"].to(dev), b["pos"].to(dev), b["days"].to(dev), b["pad"].to(dev))
            pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
            for pid, p in zip(b["id"], pred):
                np.save(os.path.join(out_dir, f"{pid}.npy"), p); n += 1
    print(f"saved {n} maps to {out_dir}")


def compare(a):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    colors = plt.get_cmap("tab20")(np.arange(20))
    colors[0] = (0.15, 0.15, 0.15, 1)   # background: dark grey
    colors[19] = (1, 1, 1, 1)           # void: white
    cmap = ListedColormap(colors)

    cols = ["Truth"] + [os.path.basename(r.rstrip("/")) for r in a.runs]
    fig, ax = plt.subplots(len(a.ids), len(cols), figsize=(2.6 * len(cols), 2.6 * len(a.ids)),
                           squeeze=False)
    used = set()
    for i, pid in enumerate(a.ids):
        y = np.load(os.path.join(a.pastis, "ANNOTATIONS", f"TARGET_{pid}.npy"))[0]
        maps = [y] + [np.load(os.path.join(r, "preds", f"{pid}.npy")) for r in a.runs]
        for j, m in enumerate(maps):
            ax[i, j].imshow(m, cmap=cmap, vmin=0, vmax=19, interpolation="nearest")
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if j > 0:  # pixel accuracy on labelled pixels, as a quick hint
                k = y != 19
                ax[i, j].set_xlabel(f"acc {100 * (m[k] == y[k]).mean():.0f}%", fontsize=8)
            used |= set(np.unique(m).tolist())
            if i == 0:
                ax[i, j].set_title(cols[j], fontsize=9)
        ax[i, 0].set_ylabel(str(pid), fontsize=9)
    fig.legend(handles=[Patch(color=colors[c], label=CLASSES[c]) for c in sorted(used)],
               loc="lower center", ncol=5, fontsize=7, frameon=False)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(a.out, dpi=130); print("saved", a.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("save")
    s.add_argument("run"); s.add_argument("--pastis"); s.add_argument("--feat_dir")
    s.add_argument("--folds", type=int, nargs="+", default=[5]); s.add_argument("--workers", type=int, default=4)
    c = sub.add_parser("compare")
    c.add_argument("runs", nargs="+"); c.add_argument("--ids", nargs="+", required=True)
    c.add_argument("--pastis", required=True); c.add_argument("--out", default="compare.png")
    a = ap.parse_args()
    save(a) if a.cmd == "save" else compare(a)


if __name__ == "__main__":
    main()
