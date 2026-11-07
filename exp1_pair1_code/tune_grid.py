"""Sweep learning rate x epoch budget for ltae / all dates / date encoding, seed 0.

OneCycleLR is stretched over --epochs, so a 40-epoch run is not a prefix of an
80-epoch run. Each pair is trained separately. Cells go to their own folder
under --out (default tune_grid/), because train.py names every run
ltae_all_date_s0 and skips a folder that already has result.json.

The manual run in --reuse (default tune_lr1e3/) is only read. If it has
finished at lr 1e-3 and 60 epochs, that result is copied into the matching
cell. Nothing is written back into the manual folder.

  python tune_grid.py --pastis /data/PASTIS --feat_dir /data/feats
  python tune_grid.py ... --list

The pair to freeze is the one with the highest fold-4 mIoU.
"""
import argparse, json, os, shutil, subprocess, sys, time

LRS = [3e-4, 6e-4, 1e-3, 3e-3, 6e-3]
EPOCHS = [40, 60, 80]
RUN = "ltae_all_date_s0"


def lr_tag(lr):
    mant, exp = f"{lr:.0e}".split("e")
    return f"{int(mant)}e{int(exp)}"


def cell_name(lr, epochs):
    return f"lr{lr_tag(lr)}_ep{epochs}"


def cells():
    return [(lr, ep) for lr in LRS for ep in EPOCHS]


def result_path(out, lr, epochs):
    return os.path.join(out, cell_name(lr, epochs), RUN, "result.json")


def finished(path, lr, epochs):
    if not os.path.exists(path):
        return None
    r = json.load(open(path))
    a = r["args"]
    ok = (abs(a["lr"] - lr) < 1e-12 and a["epochs"] == epochs and a["seed"] == 0
          and a["agg"] == "ltae" and a["dates"] == "all" and a["pe"] == "date")
    if not ok:
        sys.exit(f"{path} is a different configuration; refusing to reuse it")
    return r


def reuse_manual(out, reuse):
    """Copy a finished manual 1e-3 / 60 run into its cell. Never writes to reuse/."""
    src = os.path.join(reuse, RUN, "result.json")
    if not os.path.exists(src):
        if os.path.isdir(reuse):
            print(f"{reuse}/ has no result.json yet. Leaving it untouched.")
        return
    r = finished(src, 1e-3, 60)
    dest = os.path.join(out, cell_name(1e-3, 60), RUN)
    if os.path.exists(os.path.join(dest, "result.json")):
        print(f"keep existing {cell_name(1e-3, 60)}")
        return
    os.makedirs(dest, exist_ok=True)
    for name in ("result.json", "best.pt"):
        s = os.path.join(reuse, RUN, name)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dest, name))
    print(f"copied finished {reuse}/{RUN} -> {dest} (source not modified)")


def report(out):
    rows = []
    for lr, ep in cells():
        r = finished(result_path(out, lr, ep), lr, ep)
        if r is None:
            print(f"{cell_name(lr, ep):16s}  missing")
            continue
        best = max(r["history"], key=lambda h: h["val_mIoU"])
        rows.append((r["val"]["mIoU"], lr, ep, best["epoch"], r["test"]["mIoU"]))
    if not rows:
        return
    rows.sort(key=lambda t: (-t[0], t[2], t[1]))
    print("\nfold-4 val mIoU   (test mIoU is not used to choose)")
    print(f"{'lr':>8} {'epochs':>7} {'best_ep':>8} {'val':>8} {'test':>8}")
    for val, lr, ep, best_ep, test in rows:
        print(f"{lr_tag(lr):>8} {ep:7d} {best_ep:8d} {val:8.2f} {test:8.2f}")
    val, lr, ep, best_ep, test = rows[0]
    print(f"\nfreeze: --extra '--lr {lr_tag(lr)} --epochs {ep}'"
          f"   (fold-4 mIoU {val:.2f}, best epoch {best_ep}, test {test:.2f})")
    json.dump(
        [{"lr": lr, "epochs": ep, "best_epoch": best_ep, "val_mIoU": val, "test_mIoU": test}
         for val, lr, ep, best_ep, test in rows],
        open(os.path.join(out, "summary.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pastis", required=True)
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--out", default="tune_grid")
    ap.add_argument("--reuse", default="tune_lr1e3",
                    help="finished manual run to copy from; this folder is never written")
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--per_gpu", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        for lr, ep in cells():
            print(cell_name(lr, ep))
        print(len(cells()), "runs")
        return

    if os.path.abspath(a.out) == os.path.abspath(a.reuse):
        sys.exit("--out and --reuse must be different folders")

    reuse_manual(a.out, a.reuse)
    todo = [(lr, ep) for lr, ep in cells() if finished(result_path(a.out, lr, ep), lr, ep) is None]
    if not todo:
        print("every cell already has result.json")
        report(a.out)
        return

    if a.gpus is None:
        import torch
        gpus = list(range(torch.cuda.device_count()))
    else:
        gpus = [int(g) for g in a.gpus.split(",")]
    if not gpus:
        sys.exit("no GPU visible")
    slots = [g for g in gpus for _ in range(a.per_gpu)]
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)

    running = {}
    while todo or running:
        for slot, g in enumerate(slots):
            p = running.get(slot)
            if p is not None and p.poll() is not None:
                if p.returncode != 0:
                    print("FAILED:", p.args)
                running.pop(slot)
            if slot not in running and todo:
                lr, ep = todo.pop(0)
                cmd = (f"{sys.executable} train.py --pastis {a.pastis} --feat_dir {a.feat_dir} "
                       f"--out {os.path.join(a.out, cell_name(lr, ep))} "
                       f"--agg ltae --dates all --pe date --seed 0 "
                       f"--lr {lr_tag(lr)} --epochs {ep} --workers {a.workers}")
                log = open(os.path.join(a.out, "logs", cell_name(lr, ep) + ".log"), "w")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                running[slot] = subprocess.Popen(
                    cmd, shell=True, env=env, stdout=log, stderr=subprocess.STDOUT)
                print(f"GPU {g}: {cell_name(lr, ep)}")
        time.sleep(5)
    print("sweep done")
    report(a.out)


if __name__ == "__main__":
    main()
