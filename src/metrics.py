"""Metrics: confusion matrix, mIoU, overall accuracy, field-level accuracy, bootstrap CIs."""
import numpy as np

from .pastis import NUM_CLASSES, VOID_LABEL


def confusion(pred, target, num_classes=NUM_CLASSES, ignore=VOID_LABEL):
    """pred, target: int arrays of the same shape. Returns (C, C) counts, rows = true class."""
    m = target != ignore
    t, p = target[m].astype(np.int64), pred[m].astype(np.int64)
    return np.bincount(t * num_classes + p, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def scores(cm):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - tp
    # Same as the U-TAE code: IoU = TP / (TP + FP + FN), averaged over classes with TP+FP+FN > 0
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    return {
        "miou": float(np.nanmean(iou)) * 100,
        "oa": float(tp.sum() / max(cm.sum(), 1)) * 100,
        "iou_per_class": [None if np.isnan(v) else float(v) * 100 for v in iou],
    }


def field_hits(pred, target, parcels):
    """For each parcel: majority predicted class vs its true class. Returns (n_correct, n_fields)."""
    correct = total = 0
    for pid in np.unique(parcels):
        if pid == 0:
            continue
        m = parcels == pid
        true = np.bincount(target[m]).argmax()
        if true == VOID_LABEL or true == 0:
            continue
        maj = np.bincount(pred[m], minlength=NUM_CLASSES).argmax()
        correct += int(maj == true)
        total += 1
    return correct, total


def bootstrap_gap_difference(a_real, a_shuf, b_real, b_shuf, n_boot=1000, seed=0):
    """CI of [mIoU(a_real) - mIoU(a_shuf)] - [mIoU(b_real) - mIoU(b_shuf)], paired over patches.
    Answers: does feature set A lose more than B when the dates are shuffled?"""
    rng = np.random.RandomState(seed)
    P = a_real.shape[0]

    def gap(idx):
        m = [scores(c[idx].sum(0))["miou"] for c in (a_real, a_shuf, b_real, b_shuf)]
        return (m[0] - m[1]) - (m[2] - m[3])

    vals = [gap(rng.randint(0, P, P)) for _ in range(n_boot)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"point": float(gap(np.arange(P))), "ci95": [float(lo), float(hi)]}


def bootstrap_miou(cms, n_boot=1000, seed=0, other=None):
    """Bootstrap over test patches. cms: (P, C, C) per-patch confusion matrices.

    If `other` (same shape, same patch order) is given, returns the CI of the
    paired difference mIoU(cms) - mIoU(other).
    """
    rng = np.random.RandomState(seed)
    P = cms.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, P, P)
        a = scores(cms[idx].sum(0))["miou"]
        if other is not None:
            a -= scores(other[idx].sum(0))["miou"]
        vals.append(a)
    point = scores(cms.sum(0))["miou"] - (scores(other.sum(0))["miou"] if other is not None else 0)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"point": float(point), "ci95": [float(lo), float(hi)]}
