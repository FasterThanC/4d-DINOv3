"""NDVI probe: do the saved features follow crop growth over the season?

For every field (PASTIS parcel) and date we average the field's tokens (weighted by how much
of each 8 x 8-pixel token lies inside the field) and compute the field's true NDVI
(NIR - red) / (NIR + red) from the raw bands. Then, for each feature set:

 1. Linear probe: ridge regression features -> NDVI, trained on fields of folds 1-3,
    R^2 on fields of fold 5. Controls: mean RGB of the field (what an RGB-only model
    could know at best with a linear map) and all 10 bands (upper bound).
 2. Season figure: median true NDVI and median predicted NDVI per month for 4 common crops.
 3. Grouping without labels: each test field's monthly feature trajectory -> PCA -> k-means;
    agreement with the crop labels (ARI, NMI). NDVI trajectories alone as a reference.

    python scripts/ndvi_probe.py --pastis $PASTIS --splits splits.json --out $RUNS/probe \
        --feats dinov3=$FEATS/dinov3_sat student1=$FEATS/student1_copy student2=$FEATS/student2_rebuild
"""
import argparse
import os
import sys
from datetime import timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pastis import (BLUE_IDX, CLASS_NAMES, NIR_IDX, RED_IDX, REF_DATE, RGB_IDX,  # noqa: E402
                        load_metadata, load_parcels, load_s2, load_splits, load_target)
from src.utils import save_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pastis", required=True)
ap.add_argument("--splits", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--feats", nargs="+", required=True, help="name=folder pairs")
ap.add_argument("--train-patches", type=int, default=300)
ap.add_argument("--test-patches", type=int, default=300)
ap.add_argument("--max-fields", type=int, default=20, help="fields per patch")
ap.add_argument("--min-pixels", type=int, default=64)
ap.add_argument("--cloud", type=float, default=0.2, help="skip field-dates with mean blue reflectance above this")
ap.add_argument("--min-signal", type=float, default=0.05,
                help="skip field-dates with mean NIR + red below this (cloud shadow, no data)")
ap.add_argument("--max-train-rows", type=int, default=150000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, r2_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

os.makedirs(a.out, exist_ok=True)
rng = np.random.RandomState(a.seed)
meta = load_metadata(a.pastis)
splits = load_splits(a.splits)
feat_sets = dict(f.split("=", 1) for f in a.feats)


def pick(ids, n):
    ids = list(ids)
    return ids if (n <= 0 or n >= len(ids)) else [ids[i] for i in sorted(rng.choice(len(ids), n, replace=False))]


def month_of(day):
    d = REF_DATE + timedelta(days=int(day))
    return (d.year - 2018) * 12 + d.month - 9          # 0 = Sept 2018


def collect(ids):
    """Rows = (field, date). Returns dict of arrays + per-feature-set matrices."""
    rows = {"field": [], "crop": [], "month": [], "ndvi": [], "rgb": [], "bands": [], "clear": []}
    feats = {k: [] for k in feat_sets}
    fid = 0
    for pid in ids:
        s2 = np.asarray(load_s2(a.pastis, pid), dtype=np.float32) / 10000.0     # T, 10, H, W
        y, parc = load_target(a.pastis, pid), load_parcels(a.pastis, pid)
        loaded = {k: np.load(os.path.join(v, f"{pid}.npy"), mmap_mode="r") for k, v in feat_sets.items()}
        fields = []
        for p in np.unique(parc):
            if p == 0:
                continue
            m = parc == p
            crop = np.bincount(y[m]).argmax()
            if m.sum() >= a.min_pixels and 1 <= crop <= 18:
                fields.append((p, m, crop))
        if len(fields) > a.max_fields:
            fields = [fields[i] for i in rng.choice(len(fields), a.max_fields, replace=False)]
        days = meta[pid]["days"]
        if not fields:
            continue
        for k, f in loaded.items():
            # one weighted average per field: weight = share of each token's pixels inside the field
            assert f.shape[0] == len(days), f"patch {pid}: features and metadata disagree on T"
            h = f.shape[-1]
            s = parc.shape[0] // h
            W = np.stack([m.reshape(h, s, h, s).mean(axis=(1, 3)).ravel() for _, m, _ in fields])
            W = W / W.sum(1, keepdims=True)                                      # n_fields, h*w
            F_ = np.asarray(f, dtype=np.float32).reshape(f.shape[0], f.shape[1], -1)   # T, D, h*w
            per_field = np.einsum("tdk,fk->ftd", F_, W).astype(np.float16)      # n_fields, T, D
            feats[k].extend(list(per_field))
        for p, m, crop in fields:
            px = s2[:, :, m]                                                   # T, 10, n
            # NDVI of the field from its mean NIR and red. (A mean of per-pixel NDVI breaks on
            # PASTIS: some dark pixels have NIR + red close to 0 or below 0, which gives huge values.)
            nir, red = px[:, NIR_IDX].mean(1), px[:, RED_IDX].mean(1)
            ok = (nir + red) > a.min_signal                                     # not shadow / no data
            ndvi = np.clip((nir - red) / np.where(ok, nir + red, 1.0), -1, 1)
            rows["field"] += [fid] * len(days)
            rows["crop"] += [crop] * len(days)
            rows["month"] += [month_of(d) for d in days]
            rows["ndvi"].append(ndvi)
            rows["rgb"].append(px[:, RGB_IDX].mean(2))
            rows["bands"].append(px.mean(2))
            rows["clear"].append((px[:, BLUE_IDX].mean(1) <= a.cloud) & ok)
            fid += 1
    out = {k: np.array(v) for k, v in rows.items() if k in ("field", "crop", "month")}
    for k in ("ndvi", "clear"):
        out[k] = np.concatenate(rows[k])
    for k in ("rgb", "bands"):
        out[k] = np.concatenate(rows[k]).astype(np.float32)
    out["feats"] = {k: np.concatenate(v).astype(np.float32) for k, v in feats.items()}
    return out


train_ids, test_ids = pick(splits["train"], a.train_patches), pick(splits["test"], a.test_patches)
print(f"collecting fields from {len(train_ids)} train and {len(test_ids)} test patches ...", flush=True)
tr, te = collect(train_ids), collect(test_ids)
mtr, mte = tr["clear"], te["clear"]
print(f"train rows {mtr.sum()}, test rows {mte.sum()} (cloud-free field-dates)")
q = np.percentile(te["ndvi"][mte], [1, 25, 50, 75, 99])
print("test NDVI percentiles 1/25/50/75/99:", np.round(q, 3))

sub = np.where(mtr)[0]
if len(sub) > a.max_train_rows:
    sub = rng.choice(sub, a.max_train_rows, replace=False)

results = {"linear_probe_r2": {}, "grouping": {}}
preds = {}
inputs = {"control_rgb_mean": (tr["rgb"], te["rgb"]), "control_all_bands": (tr["bands"], te["bands"])}
inputs.update({k: (tr["feats"][k], te["feats"][k]) for k in feat_sets})
for k, (Xtr, Xte) in inputs.items():
    sc = StandardScaler().fit(Xtr[sub])
    reg = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(Xtr[sub]), tr["ndvi"][sub])
    p = reg.predict(sc.transform(Xte))
    preds[k] = p
    results["linear_probe_r2"][k] = float(r2_score(te["ndvi"][mte], p[mte]))
    print(f"NDVI R^2 on fold 5  {k:20s}: {results['linear_probe_r2'][k]:.3f}")
# sanity check: NDVI is a function of 2 of the 10 bands, so a linear map from the 10 field-mean bands
# must explain most of it. If not, the NDVI targets are broken.
if results["linear_probe_r2"]["control_all_bands"] < 0.7:
    print("FAIL NDVI probe sanity check: all-bands control R^2 < 0.7, the NDVI targets look wrong")
    sys.exit(1)
print("PASS NDVI probe sanity check (all-bands control R^2 >= 0.7)")

# 2) season figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    crops, counts = np.unique(te["crop"][mte], return_counts=True)
    top = crops[np.argsort(-counts)][:4]
    months = np.arange(0, 15)
    fig, axes = plt.subplots(1, len(top), figsize=(3.2 * len(top), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, top):
        sel = mte & (te["crop"] == c)
        for k, style in [("true", "k-")] + [(n, "--") for n in feat_sets]:
            vals = te["ndvi"] if k == "true" else preds[k]
            med = [np.median(vals[sel & (te["month"] == mm)]) if (sel & (te["month"] == mm)).any() else np.nan
                   for mm in months]
            ax.plot(months, med, style, label="true NDVI" if k == "true" else k)
        ax.set_title(CLASS_NAMES[c], fontsize=9)
        ax.set_xticks([0, 4, 8, 12])
        ax.set_xticklabels(["Sep 18", "Jan 19", "May 19", "Sep 19"], fontsize=7)
    axes[0].set_ylabel("NDVI (median over fields)")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_ndvi_season.png"), dpi=200)
    print("saved fig_ndvi_season.png")
except Exception as e:  # plotting must never kill the probe
    print("figure skipped:", e)

# 3) grouping fields by their monthly trajectories (no labels)


def trajectories(d, X, months=range(1, 13)):
    """(n_fields, len(months) * dim): monthly mean of clear dates, gaps filled from neighbours."""
    n = d["field"].max() + 1
    M = list(months)
    T = np.full((n, len(M), X.shape[1]), np.nan, dtype=np.float32)
    for j, mm in enumerate(M):
        sel = d["clear"] & (d["month"] == mm)
        if not sel.any():
            continue
        f = d["field"][sel]
        sums = np.zeros((n, X.shape[1]), dtype=np.float64)
        np.add.at(sums, f, X[sel])
        cnt = np.bincount(f, minlength=n)[:, None]
        T[:, j] = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
    for j in range(1, len(M)):                      # forward fill, then backward fill
        T[:, j] = np.where(np.isnan(T[:, j]), T[:, j - 1], T[:, j])
    for j in range(len(M) - 2, -1, -1):
        T[:, j] = np.where(np.isnan(T[:, j]), T[:, j + 1], T[:, j])
    T = np.nan_to_num(T)
    crop = np.zeros(n, dtype=int)
    crop[d["field"]] = d["crop"]
    return T, crop


grp_inputs = {"ndvi_only": (tr["ndvi"][:, None], te["ndvi"][:, None])}
grp_inputs.update({k: (tr["feats"][k], te["feats"][k]) for k in feat_sets})
for k, (Xtr, Xte) in grp_inputs.items():
    if Xtr.shape[1] > 16:
        pca = PCA(16, random_state=a.seed).fit(Xtr[sub])
        Xte = pca.transform(Xte)
    T, crop = trajectories(te, Xte)
    Z = StandardScaler().fit_transform(T.reshape(len(T), -1))
    kk = len(np.unique(crop))
    lab = KMeans(kk, n_init=10, random_state=a.seed).fit_predict(Z)
    results["grouping"][k] = {"ARI": float(adjusted_rand_score(crop, lab)),
                              "NMI": float(normalized_mutual_info_score(crop, lab)), "k": int(kk),
                              "n_fields": int(len(crop))}
    print(f"grouping {k:20s}: ARI {results['grouping'][k]['ARI']:.3f}  NMI {results['grouping'][k]['NMI']:.3f}")

results["settings"] = vars(a)
save_json(results, os.path.join(a.out, "probe_results.json"))
print("saved", os.path.join(a.out, "probe_results.json"))
