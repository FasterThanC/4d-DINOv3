"""Train one configuration on folds 1-3, pick the best epoch on fold 4, test on fold 5.

Example:
  python train.py --pastis /data/PASTIS --feat_dir /data/feats --out runs \
      --agg ltae --dates clear --pe date --seed 0
"""
import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import PastisFeatures, collate, NUM_CLASSES, VOID
from models import TemporalSegmenter


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pastis", required=True)
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--out", default="runs")
    # what the model sees
    ap.add_argument("--agg", choices=["single", "mean", "ltae"], default="ltae")
    ap.add_argument("--dates", choices=["all", "clear"], default="all")
    ap.add_argument("--pe", choices=["none", "position", "date", "both", "shuffled"], default="date")
    ap.add_argument("--month", type=int, default=6, help="for agg=single")
    ap.add_argument("--cloudy_frac", type=float, default=0.2, help="date is cloudy above this")
    ap.add_argument("--min_dates", type=int, default=8, help="keep at least this many dates")
    ap.add_argument("--pos_scale", type=float, default=10.0,
                    help="position k is encoded as k*pos_scale (~mean gap in days)")
    ap.add_argument("--label_frac", type=float, default=1.0)
    # training
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save_attn", action="store_true", help="dump L-TAE attention on the test fold")
    a = ap.parse_args(argv)
    if a.agg != "ltae":
        a.pe = "none"
    return a


def run_name(a):
    if a.agg == "single":
        n = f"single_m{a.month:02d}"
    elif a.agg == "mean":
        n = f"mean_{a.dates}"
    else:
        n = f"ltae_{a.dates}_{a.pe}"
    if a.label_frac < 1:
        n += f"_lab{int(a.label_frac * 100)}"
    return f"{n}_s{a.seed}"


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def worker_init(wid):
    # different, but reproducible, shuffles per worker/epoch for pe=shuffled
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)


class Confusion:
    def __init__(self, n=NUM_CLASSES):
        self.m = torch.zeros(n, n, dtype=torch.long)

    def add(self, pred, y):
        k = y != VOID
        self.m += torch.bincount(y[k] * NUM_CLASSES + pred[k],
                                 minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES).cpu()

    def scores(self):
        m = self.m[:VOID, :VOID].double()           # classes 0..18, void ignored
        tp = m.diag(); iou = tp / (m.sum(0) + m.sum(1) - tp).clamp(min=1)
        present = m.sum(1) > 0
        return {"mIoU": 100 * iou[present].mean().item(), "OA": 100 * (tp.sum() / m.sum()).item(),
                "IoU_per_class": [round(100 * v, 2) for v in iou.tolist()]}


@torch.no_grad()
def evaluate(model, loader, dev, attn_out=None):
    model.eval(); cm = Confusion(); dump = []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits, attn = model(b["x"].to(dev, non_blocking=True), b["pos"].to(dev),
                                 b["days"].to(dev), b["pad"].to(dev))
        cm.add(logits.argmax(1), b["y"].to(dev))
        if attn_out is not None and attn is not None:
            a = attn.float().mean(dim=(0, 3, 4)).cpu()      # [heads,B,T,h,w] -> mean over heads and patches: [B,T]
            for i, pid in enumerate(b["id"]):
                keep = ~b["pad"][i]
                dump.append({"id": pid, "attn": a[i][keep].tolist(),
                             "days": b["days"][i][keep].tolist(), "cloud": b["cloud"][i][keep].tolist()})
    if attn_out is not None and dump:
        json.dump(dump, open(attn_out, "w"))
    return cm.scores()


def main(argv=None):
    a = get_args(argv)
    seed_all(a.seed)
    name = run_name(a); out_dir = os.path.join(a.out, name); os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "result.json")):
        print("done already:", name); return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mk = lambda folds, **kw: PastisFeatures(a.pastis, a.feat_dir, folds, a, **kw)
    train_ds = mk([1, 2, 3], label_frac=a.label_frac)
    dl = lambda ds, sh: DataLoader(ds, batch_size=a.bs, shuffle=sh, num_workers=a.workers,
                                   collate_fn=collate, pin_memory=True, worker_init_fn=worker_init,
                                   persistent_workers=a.workers > 0, drop_last=sh)
    tr, va, te = dl(train_ds, True), dl(mk([4]), False), dl(mk([5]), False)

    model = TemporalSegmenter(agg=a.agg, pe=a.pe).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(tr),
                                                pct_start=0.1)
    best, hist = -1, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0
        for b in tr:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits, _ = model(b["x"].to(dev, non_blocking=True), b["pos"].to(dev),
                                  b["days"].to(dev), b["pad"].to(dev))
                loss = F.cross_entropy(logits.float(), b["y"].to(dev), ignore_index=VOID)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        v = evaluate(model, va, dev)
        hist.append({"epoch": ep, "loss": tot / len(tr), "val_mIoU": v["mIoU"], "sec": time.time() - t0})
        print(f"[{name}] ep {ep} loss {tot/len(tr):.3f} val mIoU {v['mIoU']:.2f} ({time.time()-t0:.0f}s)")
        if v["mIoU"] > best:
            best = v["mIoU"]; torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))

    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=dev))
    val = evaluate(model, va, dev)
    test = evaluate(model, te, dev,
                    attn_out=os.path.join(out_dir, "attn_test.json") if a.save_attn else None)
    res = {"name": name, "args": vars(a), "val": val, "test": test, "history": hist}
    json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
    print(f"[{name}] VAL mIoU {val['mIoU']:.2f} | TEST mIoU {test['mIoU']:.2f} OA {test['OA']:.2f}")


if __name__ == "__main__":
    main()
