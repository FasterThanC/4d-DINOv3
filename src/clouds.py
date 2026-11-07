"""Cloud share per image, used to pick cloud-free dates and a single date per series.

An image's cloud share = fraction of its pixels with blue reflectance (B2) above 2000
(0.2 x 10000), the same simple test used for the RGB scaling. Shadows are not detected.
Computed once from DATA_S2 and saved as an .npz (one array per series).
"""
import os
from datetime import date, timedelta

import numpy as np

from .pastis import BLUE_IDX, REF_DATE, load_s2

CLOUD_BLUE = 2000


def cloud_fractions(pastis_dir, meta, path):
    """{pid: (T,) float32 cloud share}; computed and saved to `path` if it does not exist."""
    if os.path.exists(path):
        d = np.load(path)
        out = {int(k): d[k] for k in d.files}
        if all(p in out for p in meta):
            return out
    out = {}
    for pid in meta:
        blue = np.asarray(load_s2(pastis_dir, pid)[:, BLUE_IDX], dtype=np.float32)   # T, H, W
        out[pid] = (blue > CLOUD_BLUE).mean(axis=(1, 2)).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path[:-4] + f".tmp{os.getpid()}.npz"
    np.savez(tmp, **{str(k): v for k, v in out.items()})
    os.replace(tmp, path)                                # appears only when complete
    return out


def month_index(day):
    """0 = Sept 2018, 1 = Oct 2018, ..., 14 = Nov 2019."""
    d = REF_DATE + timedelta(days=int(day))
    return (d.year - 2018) * 12 + d.month - 9


def month_name(m):
    y, mo = 2018 + (8 + m) // 12, (8 + m) % 12 + 1
    return date(y, mo, 1).strftime("%b %Y")


def select_dates(days, clouds, mode, month=None, max_cloud=0.1, min_dates=1):
    """Indices of the dates to keep for one series (sorted in time).

    all       : every date
    cloudfree : dates with cloud share <= max_cloud; if fewer than min_dates, the min_dates
                least cloudy dates (Experiment 1 design: max_cloud 0.2, min_dates 8)
    single    : one date in `month` (0 = Sept 2018 ... 14 = Nov 2019, the least cloudy one);
                if the series has no image that month, the image closest to the middle of it
    calmonth  : one date in calendar month `month` (1 = Jan ... 12 = Dec; Sep and Oct exist in
                2018 and 2019), the least cloudy one; if none, the image closest to the 15th
                of that month (2018 for Sep-Dec, else 2019). Experiment 1 design.
    """
    T = len(days)
    if mode == "all":
        return np.arange(T)
    if mode == "cloudfree":
        keep = np.where(clouds <= max_cloud)[0]
        if len(keep) < min_dates:
            keep = np.sort(np.argsort(clouds, kind="stable")[:min_dates])
        return keep if len(keep) else np.array([int(np.argmin(clouds))])
    if mode == "calmonth":
        cal = np.array([(REF_DATE + timedelta(days=int(d))).month for d in days])
        inm = np.where(cal == month)[0]
        if len(inm):
            return np.array([inm[np.argmin(clouds[inm])]])
        mid = (date(2018 if month >= 9 else 2019, month, 15) - REF_DATE).days
        return np.array([int(np.argmin(np.abs(np.asarray(days) - mid)))])
    if mode == "single":
        months = np.array([month_index(d) for d in days])
        inm = np.where(months == month)[0]
        if len(inm):
            return np.array([inm[np.argmin(clouds[inm])]])
        y, mo = 2018 + (8 + month) // 12, (8 + month) % 12 + 1
        mid = (date(y, mo, 15) - REF_DATE).days
        return np.array([int(np.argmin(np.abs(np.asarray(days) - mid)))])
    raise ValueError(mode)
