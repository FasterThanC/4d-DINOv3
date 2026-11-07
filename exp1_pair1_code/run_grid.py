"""Run the Experiment 1 grid across all GPUs of this machine, sharded across machines.

  # machine 0 of 3:
  python run_grid.py --pastis /data/PASTIS --feat_dir /data/feats --shard 0 --nshards 3
  # machine 1 of 3: ... --shard 1 --nshards 3   (etc.)
  python run_grid.py ... --list      # just print the runs

Finished runs (result.json exists) are skipped, so re-launching is safe.
"""
import argparse, os, subprocess, sys, time, itertools

SEEDS = [0, 1, 2]


def best_month(out):
    """Month with the best fold-4 mIoU among the seed-0 single-date runs."""
    import glob, json
    scores = {}
    for f in glob.glob(os.path.join(out, "single_m*_s0", "result.json")):
        r = json.load(open(f)); scores[r["args"]["month"]] = r["val"]["mIoU"]
    if len(scores) < 12:
        sys.exit(f"only {len(scores)}/12 single-date months finished; run the main grid first")
    m = max(scores, key=scores.get)
    print("best single-date month on fold 4:", m, f"({scores[m]:.1f} mIoU)")
    return m


def grid(single_best=None):
    if single_best is not None:                                  # stage 2: extra seeds, best month
        return [f"--agg single --month {single_best} --seed {s}" for s in SEEDS[1:]]
    runs = [f"--agg single --month {m} --seed 0" for m in range(1, 13)]  # stage 1: every month, 1 seed
    for s in SEEDS:
        for d in ("all", "clear"):
            runs.append(f"--agg mean --dates {d} --seed {s}")
        # main 2 x 4 grid
        for d, pe in itertools.product(("all", "clear"), ("none", "position", "date", "both")):
            runs.append(f"--agg ltae --dates {d} --pe {pe} --seed {s} --save_attn")
    # optional: shuffled dates, one seed (appendix)
    runs.append("--agg ltae --dates all --pe shuffled --seed 0")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pastis", required=True); ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--gpus", default=None, help="e.g. 0,1,2,3 (default: all visible)")
    ap.add_argument("--per_gpu", type=int, default=2, help="runs at once per GPU (they are small)")
    ap.add_argument("--extra", default="", help="extra args for train.py, e.g. '--epochs 40'")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--single_best", action="store_true",
                    help="stage 2: run seeds 1-2 for the best single-date month")
    a = ap.parse_args()

    runs = grid(best_month(a.out) if a.single_best else None)[a.shard::a.nshards]
    if a.list:
        print("\n".join(runs)); print(len(runs), "runs in this shard"); return

    if a.gpus is None:
        import torch; gpus = list(range(torch.cuda.device_count()))
    else:
        gpus = [int(g) for g in a.gpus.split(",")]
    slots = [g for g in gpus for _ in range(a.per_gpu)]
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    todo, running = list(runs), {}
    while todo or running:
        for slot, g in enumerate(slots):
            p = running.get(slot)
            if p is not None and p.poll() is not None:
                if p.returncode != 0:
                    print("FAILED:", p.args[-1] if isinstance(p.args, list) else p.args)
                running.pop(slot)
            if slot not in running and todo:
                r = todo.pop(0)
                cmd = (f"{sys.executable} train.py --pastis {a.pastis} --feat_dir {a.feat_dir} "
                       f"--out {a.out} {r} {a.extra}")
                log = open(os.path.join(a.out, "logs", r.replace(" ", "").replace("--", "_") + ".log"), "w")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                running[slot] = subprocess.Popen(cmd, shell=True, env=env, stdout=log, stderr=subprocess.STDOUT)
                print(f"GPU {g}: {r}")
        time.sleep(5)
    print("shard done")


if __name__ == "__main__":
    main()
