"""Per-date cloud fraction for every PASTIS series (run once, before training).

Rule (fixed for every experiment, write it in the report):
  a pixel is "cloud" if its blue band (B2) reflectance > BLUE_THRESHOLD
  a date is "cloudy" if more than CLOUDY_DATE_FRAC of its pixels are cloud

Output: <feat_dir>/<id>_cloud.npy  float32 [T], fraction of cloud pixels per date,
in the same date order as <id>_days.npy.

Usage:
  python compute_clouds.py --pastis /data/PASTIS --feat_dir /data/feats --check
"""
import argparse, json, os
import numpy as np

BLUE_THRESHOLD = 2000   # same value as cloud_blue_threshold in rgb_percentiles.json
BLUE_BAND = 0           # PASTIS S2 band order: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pastis", required=True, help="PASTIS root (has DATA_S2/, metadata.geojson)")
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--check", action="store_true",
                    help="also check that each series has as many dates as its feature file")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.pastis, "metadata.geojson")))
    ids = [f["properties"]["ID_PATCH"] for f in meta["features"]]
    fracs_all = []
    for i, pid in enumerate(ids):
        s2 = np.load(os.path.join(args.pastis, "DATA_S2", f"S2_{pid}.npy"), mmap_mode="r")  # [T,10,128,128]
        frac = (np.asarray(s2[:, BLUE_BAND]) > BLUE_THRESHOLD).mean(axis=(1, 2)).astype(np.float32)
        if args.check:
            days = np.load(os.path.join(args.feat_dir, f"{pid}_days.npy"))
            assert len(days) == len(frac), f"{pid}: {len(days)} feature dates vs {len(frac)} S2 dates"
        np.save(os.path.join(args.feat_dir, f"{pid}_cloud.npy"), frac)
        fracs_all.append(frac)
        if i % 200 == 0:
            print(f"{i}/{len(ids)}")

    f = np.concatenate(fracs_all)
    print(f"dates: {len(f)}")
    for t in (0.05, 0.1, 0.2, 0.3, 0.5, 0.9):
        print(f"  share of dates with cloud fraction > {t:.2f}: {(f > t).mean():.3f}")


if __name__ == "__main__":
    main()
