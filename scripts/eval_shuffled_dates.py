"""Test-time date shuffling: how much does a model TRAINED WITH REAL DATES rely on them?

For each finished L-TAE run (trained with real dates, all dates), the saved best checkpoint is
tested on fold 5 twice: with the real dates and with shuffled dates (each image keeps its
features but gets another image's date; the permutation is fixed per series and identical for
every model, see FeatureSeriesDataset). Nothing is trained.

The real-date score must reproduce the run's results.json (checked), so the evaluation matches
training exactly. Per-series confusion matrices are saved for paired bootstrap CIs
(scripts/date_reliance.py).

    python scripts/eval_shuffled_dates.py --pastis $PASTIS --splits $SPLITS --feats-root $FEATS \
        --out $RUNS/reliance --runs $RUNS/temporal_final/*__ltae__real__train__s*
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import FeatureSeriesDataset, pad_collate  # noqa: E402
from src.metrics import confusion, scores  # noqa: E402
from src.pastis import NUM_CLASSES, load_metadata, load_splits  # noqa: E402
from src.temporal import SegModel  # noqa: E402
from src.utils import autocast, get_device, save_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pastis", required=True)
ap.add_argument("--splits", required=True)
ap.add_argument("--feats-root", required=True, help="folder that holds the feature folders ($FEATS)")
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--device", default="auto")
a = ap.parse_args()

device = get_device(a.device)
meta = load_metadata(a.pastis)
test_ids = load_splits(a.splits)["test"]
os.makedirs(a.out, exist_ok=True)


def evaluate(model, ds):
    per, pids = [], []
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers, collate_fn=pad_collate)
    with torch.no_grad():
        for b in dl:
            with autocast(device):
                logits = model(b["x"].to(device).float(), b["days"].to(device), b["pad_mask"].to(device))
            pred = logits.argmax(1).cpu().numpy()
            for i in range(len(pred)):
                per.append(confusion(pred[i], b["y"][i].numpy()))
                pids.append(int(b["pid"][i]))
    return np.stack(per), np.array(pids)


summary = []
for run in a.runs:
    name = os.path.basename(os.path.normpath(run))
    dst = os.path.join(a.out, name + ".npz")
    res = json.load(open(os.path.join(run, "results.json")))
    args = res["args"]
    assert args["temporal"] == "ltae" and args["dates"] == "real" and args.get("select", "all") == "all", \
        f"{name}: only L-TAE runs trained with real dates on all dates"
    if os.path.exists(dst):
        d = np.load(dst)
        cr, cs = d["cms_real"], d["cms_shuffled"]
    else:
        feats = os.path.join(a.feats_root, res["feature_name"])
        ck = torch.load(os.path.join(run, "best.pt"), map_location="cpu", weights_only=False)
        model = SegModel(ck["in_dim"], temporal="ltae", num_classes=NUM_CLASSES, use_dates=True)
        model.load_state_dict(ck["model"])
        model.eval().to(device)
        cr, pr = evaluate(model, FeatureSeriesDataset(feats, a.pastis, test_ids, meta, date_mode="real"))
        cs, ps = evaluate(model, FeatureSeriesDataset(feats, a.pastis, test_ids, meta, date_mode="shuffled"))
        assert (pr == ps).all() and list(pr) == list(test_ids)
        np.savez_compressed(dst, cms_real=cr, cms_shuffled=cs, pids=pr)
    m_real, m_shuf = scores(cr.sum(0))["miou"], scores(cs.sum(0))["miou"]
    ok = abs(m_real - res["test_miou"]) < 0.05
    print(f"{name}: real {m_real:.2f} (results.json {res['test_miou']:.2f}{'' if ok else '  MISMATCH'}), "
          f"shuffled at test {m_shuf:.2f}, drop {m_real - m_shuf:.2f}", flush=True)
    assert ok, f"{name}: re-evaluation does not reproduce results.json; stop and check"
    summary.append({"run": name, "features": res["feature_name"], "test_miou_real": m_real,
                    "test_miou_shuffled_at_test": m_shuf})
save_json(summary, os.path.join(a.out, "ltae_runs.json"))
print("saved", a.out)
