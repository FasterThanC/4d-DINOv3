"""Check a folder of saved token features. Prints PASS/FAIL lines; exit code 1 on failure.

    # structure only (all series in the splits file): fast, run after every extraction
    python scripts/check_features.py --pastis $PASTIS --feats $FEATS/dinov3_sat --splits splits.json
    # + linear probe and a picture to look at (used in the preflight)
    python scripts/check_features.py ... --probe --figure out.png

Structure: every series has <id>.npy with shape (T, D, 16, 16), T = metadata dates, values finite,
<id>_days.npy equal to the metadata; tokens differ between places and between dates.

Probe (is the content useful?): logistic regression from ONE image's token to the main crop class
under that token (8 x 8 px), trained on the split's train series, tested on its test series.
It must beat "always predict the most common class". For comparison it also reports the same probe
on simple pixel statistics (mean and std of R, G, B under the token). If the tokens lose to that,
the image preparation is probably wrong (band order, scaling): look at the figure.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pastis import CLASS_NAMES, RGB_IDX, VOID_LABEL, load_metadata, load_s2, load_splits, load_target  # noqa: E402
from src.utils import load_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pastis", required=True)
ap.add_argument("--feats", required=True)
ap.add_argument("--splits", required=True)
ap.add_argument("--probe", action="store_true")
ap.add_argument("--figure", default="")
ap.add_argument("--dates-per-series", type=int, default=4)
ap.add_argument("--max-train-tokens", type=int, default=40000)
ap.add_argument("--margin", type=float, default=0.0, help="probe must beat the most common class by more than this (points)")
a = ap.parse_args()

ok = True


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
    ok &= bool(cond)


meta = load_metadata(a.pastis)
splits = load_splits(a.splits)
ids = sorted({p for k in ("train", "val", "test") for p in splits.get(k, [])})
name = os.path.basename(os.path.normpath(a.feats))
print(f"== {name}: {len(ids)} series")

missing, bad, D = [], [], None
for pid in ids:
    f = os.path.join(a.feats, f"{pid}.npy")
    fd = os.path.join(a.feats, f"{pid}_days.npy")
    if not (os.path.exists(f) and os.path.exists(fd)):
        missing.append(pid)
        continue
    x = np.load(f, mmap_mode="r")
    D = D or x.shape[1]
    if x.ndim != 4 or x.shape[0] != len(meta[pid]["days"]) or x.shape[1] != D or x.shape[2:] != (16, 16):
        bad.append((pid, x.shape))
    elif not np.array_equal(np.load(fd), meta[pid]["days"]):
        bad.append((pid, "days differ from metadata"))
check("feature + days file for every series", not missing, f"missing {len(missing)}: {missing[:5]}")
check("shape (T, D, 16, 16), T = metadata dates, days = metadata", not bad, str(bad[:3]))
if missing or bad:
    print("FEATURE CHECK FAILED")
    sys.exit(1)

rng = np.random.RandomState(0)
sample = [int(p) for p in rng.choice(ids, size=min(20, len(ids)), replace=False)]
finite, s_space, s_time, dup = True, [], [], 0
for pid in sample:
    x = np.load(os.path.join(a.feats, f"{pid}.npy")).astype(np.float32)       # T, D, 16, 16
    finite &= bool(np.isfinite(x).all())
    s_space.append(x.std(axis=(2, 3)).mean())                                  # across places
    s_time.append(x.std(axis=0).mean())                                        # across dates
    flat = x.reshape(x.shape[0], -1)
    dup += int((np.abs(np.diff(flat, axis=0)).max(1) == 0).sum())
check("values finite (no NaN / inf)", finite)
check("tokens differ between places", np.mean(s_space) > 1e-3, f"mean std {np.mean(s_space):.4f}")
check("tokens differ between dates", np.mean(s_time) > 1e-3, f"mean std {np.mean(s_time):.4f}")
if dup:
    print(f"WARN {dup} pairs of neighbouring dates have identical tokens (same image twice?)")


def token_labels(y):
    """Main class per 8 x 8 px token; -1 if more than half of the pixels are void."""
    blocks = y.reshape(16, 8, 16, 8).transpose(0, 2, 1, 3).reshape(256, 64)
    out = np.full(256, -1)
    for i, b in enumerate(blocks):
        b = b[b != VOID_LABEL]
        if len(b) >= 32:
            out[i] = np.bincount(b).argmax()
    return out


def gather(series):
    X, P, Y = [], [], []
    for pid in series:
        x = np.load(os.path.join(a.feats, f"{pid}.npy"), mmap_mode="r")
        T = x.shape[0]
        ts = np.linspace(0, T - 1, a.dates_per_series + 2)[1:-1].round().astype(int)   # spread over the season
        lab = token_labels(load_target(a.pastis, pid))
        keep = lab >= 0
        s2 = load_s2(a.pastis, pid)
        for t in ts:
            tok = np.asarray(x[t], dtype=np.float32).reshape(x.shape[1], 256).T          # 256, D
            rgb = np.asarray(s2[t][RGB_IDX], dtype=np.float32).reshape(3, 16, 8, 16, 8)
            pix = np.concatenate([rgb.mean(axis=(2, 4)), rgb.std(axis=(2, 4))]).reshape(6, 256).T
            X.append(tok[keep]), P.append(pix[keep]), Y.append(lab[keep])
    return np.concatenate(X), np.concatenate(P), np.concatenate(Y)


if a.probe:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xtr, Ptr, Ytr = gather(splits["train"])
    Xte, Pte, Yte = gather(splits["test"])
    if len(Ytr) > a.max_train_tokens:
        j = rng.choice(len(Ytr), a.max_train_tokens, replace=False)
        Xtr, Ptr, Ytr = Xtr[j], Ptr[j], Ytr[j]

    def probe(tr, te):
        sc = StandardScaler().fit(tr)
        clf = LogisticRegression(max_iter=1000, C=0.1).fit(sc.transform(tr), Ytr)
        return 100 * (clf.predict(sc.transform(te)) == Yte).mean()

    major = np.bincount(Ytr).argmax()
    acc_major = 100 * (Yte == major).mean()
    acc_tok, acc_pix = probe(Xtr, Xte), probe(Ptr, Pte)
    print(f"probe on {len(Ytr)} train / {len(Yte)} test tokens: tokens {acc_tok:.1f}%, "
          f"pixel stats {acc_pix:.1f}%, always '{CLASS_NAMES[major]}' {acc_major:.1f}%")
    # One image, one 80 m token, a linear model and only the pilot series: accuracy is low even
    # with correct inputs. Broken inputs (wrong bands, wrong scaling) make the tokens useless, so
    # the check is that they beat both "always the most common class" and simple pixel statistics.
    check(f"probe beats 'always the most common class' (by > {a.margin} points)",
          acc_tok > acc_major + a.margin, f"{acc_tok:.1f} vs {acc_major:.1f}")
    check("probe beats the same probe on pixel statistics (mean/std of R, G, B)",
          acc_tok > acc_pix, f"{acc_tok:.1f} vs {acc_pix:.1f}")

if a.figure:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    man_path = os.path.join(a.feats, "manifest.json")
    pct = load_json(man_path).get("percentiles") if os.path.exists(man_path) else None
    show = splits["test"][:3]
    fig, ax = plt.subplots(len(show), 7, figsize=(17, 2.6 * len(show)), squeeze=False)
    for r, pid in enumerate(show):
        x = np.load(os.path.join(a.feats, f"{pid}.npy"), mmap_mode="r")
        s2 = load_s2(a.pastis, pid)
        ts = np.linspace(0, x.shape[0] - 1, 5)[1:-1].round().astype(int)
        pcs = []
        for t in ts:                                   # main directions per image: shows the layout
            tok = np.asarray(x[t], dtype=np.float32).reshape(x.shape[1], 256).T
            tok = tok - tok.mean(0)
            v = np.linalg.svd(tok, full_matrices=False)[2][:3]
            pc = (tok @ v.T).reshape(16, 16, 3)
            pcs.append((pc - pc.min((0, 1))) / (np.ptp(pc, axis=(0, 1)) + 1e-6))
        for c, t in enumerate(ts):
            rgb = np.asarray(s2[t][RGB_IDX], dtype=np.float32).transpose(1, 2, 0)
            if pct:
                rgb = (rgb - np.array(pct["low"])) / (np.array(pct["high"]) - np.array(pct["low"]))
            else:
                rgb = rgb / np.percentile(rgb, 99)
            ax[r, c].imshow(rgb.clip(0, 1))
            ax[r, c].set_title(f"{pid} RGB day {meta[pid]['days'][t]}", fontsize=8)
            ax[r, 3 + c].imshow(pcs[c], interpolation="nearest")
            ax[r, 3 + c].set_title(f"tokens (3 main directions) day {meta[pid]['days'][t]}", fontsize=8)
        ax[r, 6].imshow(load_target(a.pastis, pid), cmap="tab20", vmin=0, vmax=19, interpolation="nearest")
        ax[r, 6].set_title("crop labels", fontsize=8)
    for axx in ax.ravel():
        axx.axis("off")
    fig.suptitle(f"{name}: RGB as given to the model, tokens, labels. Fields should be visible in all three.")
    fig.tight_layout()
    fig.savefig(a.figure, dpi=90)
    print("figure:", a.figure)

print("FEATURE CHECK PASSED" if ok else "FEATURE CHECK FAILED")
sys.exit(0 if ok else 1)
