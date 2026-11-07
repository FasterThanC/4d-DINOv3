"""Does L-TAE attend to clear dates, or to certain seasons? (uses attn_test.json from a run)

  python attention_analysis.py runs/ltae_all_date_s0/attn_test.json --cloudy_frac 0.2
"""
import argparse, json, datetime as dt
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--cloudy_frac", type=float, default=0.2)
a = ap.parse_args()
rows = json.load(open(a.path))
att, cloud, days = [], [], []
for r in rows:
    w = np.array(r["attn"]); att += list(w * len(w))     # x T: 1.0 = uniform attention
    cloud += r["cloud"]; days += r["days"]
att, cloud, days = map(np.array, (att, cloud, days))
clear = cloud <= a.cloudy_frac

print(f"relative attention, clear dates:  {att[clear].mean():.2f}")
print(f"relative attention, cloudy dates: {att[~clear].mean():.2f}")
print(f"Spearman(attention, cloud fraction): "
      f"{np.corrcoef(np.argsort(np.argsort(att)), np.argsort(np.argsort(cloud)))[0,1]:.2f}")

month = np.array([(dt.date(2018, 9, 1) + dt.timedelta(int(d))).month for d in days])
order = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(cloud, att, s=2, alpha=0.2); ax[0].set_xlabel("cloud fraction"); ax[0].set_ylabel("attention × T")
ax[0].axhline(1, color="k", lw=0.5); ax[0].set_title("Attention vs clouds")
ax[1].bar(range(12), [att[clear & (month == m)].mean() if (clear & (month == m)).any() else 0 for m in order])
ax[1].set_xticks(range(12)); ax[1].set_xticklabels(["S","O","N","D","J","F","M","A","M","J","J","A"])
ax[1].axhline(1, color="k", lw=0.5); ax[1].set_title("Attention by month, clear dates only")
plt.tight_layout(); out = a.path.replace(".json", ".png"); plt.savefig(out, dpi=120); print("saved", out)
