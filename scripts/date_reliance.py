"""Table: how much each model, trained with real dates, loses when the dates are shuffled at test.

Reads the .npz files of eval_shuffled_dates.py and eval_utae_pretrained.py. Groups (all seeds pooled):
  U-TAE (official, pre-trained, raw 10 bands) | DINOv3 + L-TAE | student 1 | student 2 | student 3
For each: test mIoU with real dates, with shuffled dates, the drop with a paired bootstrap 95% CI
(over test series), and the difference of the drop to DINOv3 + L-TAE.

    python scripts/date_reliance.py --dir $RUNS/reliance --out $RUNS/reliance/date_reliance.json
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import bootstrap_gap_difference, bootstrap_miou, scores  # noqa: E402
from src.utils import save_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--labels", default="train", help="label split of the L-TAE runs to use")
ap.add_argument("--n-boot", type=int, default=1000)
a = ap.parse_args()

GROUPS = [("U-TAE (pre-trained, 10 bands)", r"^utae_pretrained$"),
          ("DINOv3 + L-TAE", rf"^dinov3_sat__ltae__real__{a.labels}__s\d+$"),
          ("Student 1 (copy) + L-TAE", rf"^student1_copy(_s\d+)?__ltae__real__{a.labels}__s\d+$"),
          ("Student 2 (copy + rebuild) + L-TAE", rf"^student2_rebuild(_s\d+)?__ltae__real__{a.labels}__s\d+$"),
          ("Student 3 (rebuild only) + L-TAE", rf"^student3_rebuild_only__ltae__real__{a.labels}__s\d+$")]

files = {os.path.basename(f)[:-4]: f for f in glob.glob(os.path.join(a.dir, "*.npz"))}
pooled, pid_ref = {}, None
for gname, pat in GROUPS:
    names = sorted(n for n in files if re.match(pat, n))
    if not names:
        print(f"(no runs for {gname})")
        continue
    cr = cs = None
    for n in names:
        d = np.load(files[n])
        if pid_ref is None:
            pid_ref = d["pids"]
        assert (d["pids"] == pid_ref).all(), f"{n}: different test series order"
        cr = d["cms_real"].astype(np.int64) if cr is None else cr + d["cms_real"]
        cs = d["cms_shuffled"].astype(np.int64) if cs is None else cs + d["cms_shuffled"]
    pooled[gname] = (cr, cs, len(names))

rows, ref = [], "DINOv3 + L-TAE"
print(f"{'model':38s} {'runs':>4s} {'real':>6s} {'shuffled':>8s} {'drop [95% CI]':>22s} {'drop - DINOv3 drop [95% CI]':>30s}")
for gname, (cr, cs, n) in pooled.items():
    drop = bootstrap_miou(cr, n_boot=a.n_boot, other=cs)
    row = {"model": gname, "runs": n, "miou_real": scores(cr.sum(0))["miou"],
           "miou_shuffled_at_test": scores(cs.sum(0))["miou"], "drop": drop["point"], "drop_ci95": drop["ci95"]}
    extra = ""
    if gname != ref and ref in pooled:
        g = bootstrap_gap_difference(cr, cs, pooled[ref][0], pooled[ref][1], n_boot=a.n_boot)
        row["drop_minus_dinov3_drop"], row["drop_minus_dinov3_drop_ci95"] = g["point"], g["ci95"]
        extra = f"{g['point']:6.2f} [{g['ci95'][0]:.2f}, {g['ci95'][1]:.2f}]"
    rows.append(row)
    print(f"{gname:38s} {n:4d} {row['miou_real']:6.2f} {row['miou_shuffled_at_test']:8.2f} "
          f"{drop['point']:6.2f} [{drop['ci95'][0]:.2f}, {drop['ci95'][1]:.2f}] {extra:>30s}")
save_json({"labels": a.labels, "note": "models trained with real dates; dates shuffled only at test; "
           "mIoU on fold 5; CIs: paired bootstrap over test series, seeds pooled", "rows": rows}, a.out)
print("saved", a.out)
