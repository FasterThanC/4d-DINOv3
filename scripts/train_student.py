"""Experiment 2: train a 10-band student from the saved DINOv3 tokens (no labels).

    Student 1 (copy only):        --lam 0   --name student1_copy
    Student 2 (copy + rebuild):   --lam 1   --name student2_rebuild
    Optional control, rebuild only (is DINOv3 needed at all?):
                                  --lam 1 --copy-weight 0 --name student3_rebuild_only

Both heads are always trained, so the logs of every student show the same two numbers:
how close the tokens are to DINOv3 (cosine) and how well the 7 non-RGB bands can be rebuilt
from them. When a loss has weight 0, its head is trained on detached tokens: it measures
the tokens but does not change the student.

Trains on all images of folds 1-3; checks on fold 4 (places the student has not seen):
copy loss, cosine similarity to DINOv3, rebuild error. A checkpoint is written after
every epoch, and --resume continues from the last one (useful on spot / time-limited jobs).

    python scripts/train_student.py --pastis $PASTIS --teacher $FEATS/dinov3_sat \
        --splits splits.json --out $RUNS/students --name student2_rebuild --lam 1
"""
import argparse
import math
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import ImageDistillDataset  # noqa: E402
from src.pastis import load_band_norm, load_metadata, load_splits  # noqa: E402
from src.student import Student, copy_loss, make_student_backbone, rebuild_loss  # noqa: E402
from src.utils import CSVLogger, Timer, autocast, get_device, git_commit, save_json, set_seed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pastis", required=True)
ap.add_argument("--teacher", required=True, help="folder with the saved DINOv3 tokens")
ap.add_argument("--splits", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--lam", type=float, default=0.0, help="weight of the rebuild loss (0 = copy only)")
ap.add_argument("--copy-weight", type=float, default=1.0, help="weight of the copy loss (0 = rebuild only)")
ap.add_argument("--init", default="facebook/dinov3-vits16-pretrain-lvd1689m",
                help='student start weights (hub id / folder), or "random"')
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=2e-4, help="backbone learning rate (heads use 5x)")
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--warmup", type=float, default=1.0, help="warm-up epochs")
ap.add_argument("--val-images", type=int, default=4000, help="fold-4 images used for checks")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="auto")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--max-steps", type=int, default=0, help="debug: stop after N steps")
a = ap.parse_args()
assert a.copy_weight > 0 or a.lam > 0, "at least one loss needs a weight above 0"

set_seed(a.seed)
device = get_device(a.device)
run_dir = os.path.join(a.out, a.name)
os.makedirs(run_dir, exist_ok=True)

meta = load_metadata(a.pastis)
splits = load_splits(a.splits)
band_mean, band_std = load_band_norm(a.pastis, folds=(1, 2, 3))
teacher_dim = None

train_ds = ImageDistillDataset(a.pastis, a.teacher, splits["train"], meta, band_mean, band_std)
val_ds = ImageDistillDataset(a.pastis, a.teacher, splits["val"], meta, band_mean, band_std,
                             max_items=a.val_images, seed=0)      # same check images for every run
train_check = ImageDistillDataset(a.pastis, a.teacher, splits["train"], meta, band_mean, band_std,
                                  max_items=a.val_images, seed=1)
teacher_dim = train_ds[0]["z"].shape[-1]
print(f"train images: {len(train_ds)}, val images: {len(val_ds)}, teacher dim: {teacher_dim}")

kw = dict(num_workers=a.workers, pin_memory=device.type == "cuda",
          persistent_workers=a.workers > 0)
train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, drop_last=True, **kw)
val_dl = DataLoader(val_ds, batch_size=a.batch, shuffle=False, **kw)
check_dl = DataLoader(train_check, batch_size=a.batch, shuffle=False, **kw)

backbone = make_student_backbone(a.init, in_channels=10)
model = Student(backbone, teacher_dim=teacher_dim).to(device)
n_params = sum(p.numel() for p in model.backbone.parameters())
print(f"student backbone parameters: {n_params / 1e6:.1f} M")

heads = list(model.projector.parameters()) + list(model.rebuild.parameters())
opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": a.lr},
                         {"params": heads, "lr": a.lr * 5}], weight_decay=a.wd)
steps_per_epoch = len(train_dl)
total = a.epochs * steps_per_epoch
warm = int(a.warmup * steps_per_epoch)


def lr_factor(step):
    if step < warm:
        return (step + 1) / max(warm, 1)
    return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
start_epoch, best = 0, float("inf")
last_ckpt = os.path.join(run_dir, "last.pt")
if a.resume and os.path.exists(last_ckpt):
    ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    start_epoch, best = ck["epoch"] + 1, ck["best"]
    print(f"resumed from epoch {start_epoch}")

save_json({**vars(a), "teacher_dim": teacher_dim, "student_params_M": n_params / 1e6,
           "train_images": len(train_ds), "commit": git_commit()},
          os.path.join(run_dir, "config.json"))
log = CSVLogger(os.path.join(run_dir, "log.csv"))
timer = Timer()


@torch.no_grad()
def evaluate(dl):
    model.eval()
    s = {"copy": 0.0, "cos": 0.0, "rebuild": 0.0, "rebuild_zero": 0.0, "n": 0}
    for b in dl:
        x, z, r = b["x"].to(device), b["z"].to(device), b["rebuild"].to(device)
        with autocast(device):
            _, proj, rec = model(x)
        c = copy_loss(proj, z).item()
        n = x.shape[0]
        s["copy"] += c * n
        s["cos"] += (1 - c) * n
        s["rebuild"] += rebuild_loss(rec, r).item() * n
        s["rebuild_zero"] += r.float().pow(2).mean().item() * n      # error of predicting the band average
        s["n"] += n
    model.train()
    return {k: v / s["n"] for k, v in s.items() if k != "n"}


step = start_epoch * steps_per_epoch
for epoch in range(start_epoch, a.epochs):
    model.train()
    run = {"copy": 0.0, "rebuild": 0.0, "n": 0}
    for b in train_dl:
        x, z, r = b["x"].to(device, non_blocking=True), b["z"].to(device, non_blocking=True), \
            b["rebuild"].to(device, non_blocking=True)
        with autocast(device):
            _, proj, rec = model(x, detach_copy=a.copy_weight == 0, detach_rebuild=a.lam == 0)
        lc, lr_ = copy_loss(proj, z), rebuild_loss(rec, r)
        # a head with weight 0 is still trained (weight 1) but on detached tokens -> only measures
        loss = (a.copy_weight or 1.0) * lc + (a.lam or 1.0) * lr_
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        run["copy"] += lc.item()
        run["rebuild"] += lr_.item()
        run["n"] += 1
        if step % 100 == 0:
            print(f"ep {epoch} step {step}/{total} copy {lc.item():.4f} rebuild {lr_.item():.4f}", flush=True)
        if a.max_steps and step >= a.max_steps:
            break
    v, tr = evaluate(val_dl), evaluate(check_dl)
    val_obj = a.copy_weight * v["copy"] + a.lam * v["rebuild"]
    row = {"epoch": epoch, "step": step, "train_copy": run["copy"] / max(run["n"], 1),
           "train_rebuild": run["rebuild"] / max(run["n"], 1),
           "train_cos": tr["cos"], "val_cos": v["cos"], "val_copy": v["copy"],
           "val_rebuild": v["rebuild"], "val_rebuild_zero": v["rebuild_zero"], "train_check_rebuild": tr["rebuild"],
           "val_objective": val_obj, "hours": round(timer.hours(), 3)}
    log.log(row)
    print(row, flush=True)
    state = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
             "epoch": epoch, "best": min(best, val_obj)}
    torch.save(state, last_ckpt + ".tmp")
    os.replace(last_ckpt + ".tmp", last_ckpt)
    if val_obj < best:
        best = val_obj
        torch.save({"model": model.state_dict(), "epoch": epoch, "val": v,
                    "teacher_dim": teacher_dim, "init": a.init,
                    "backbone_config": model.backbone.config.to_dict()},
                   os.path.join(run_dir, "best.pt"))
    if a.max_steps and step >= a.max_steps:
        break

save_json({"best_val_objective": best, "hours": timer.hours()}, os.path.join(run_dir, "done.json"))
print("finished", run_dir)
