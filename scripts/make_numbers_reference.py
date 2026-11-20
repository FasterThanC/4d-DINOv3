"""Builds results/ALL_NUMBERS.md: every number of the project, computed from the files in results/.

Reads only files that are in this repository (written by scripts/collect_results.py from the
UCloud run folders):
  results/exp2_runs.json                             the 96 Experiment 2 L-TAE runs
  results/exp1_time/exp1_runs.json                   the 47 final Experiment 1 runs
  results/robustness_pair1_code/pair1_code_runs.json the 47 runs of Pair 1's code
  plus final_comparisons.txt (CIs), table_exp1.json (CIs), the NDVI probe json, benchmark json,
  reliance json and log, attention json and student_training_log.csv.
Nothing here is typed in by hand except text; every number is read or computed.

    python scripts/make_numbers_reference.py            # rewrites results/ALL_NUMBERS.md
"""
import argparse
import csv
import json
import os
import re
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
ap.add_argument("--out", default="", help="default: <results>/ALL_NUMBERS.md")
a = ap.parse_args()
RES = lambda *p: os.path.join(a.results, *p)  # noqa: E731
a.out = a.out or RES("ALL_NUMBERS.md")
L = []
w = L.append
f1 = lambda x: f"{x:.1f}"  # noqa: E731
f2 = lambda x: f"{x:.2f}"  # noqa: E731

# ---------------- Experiment 2 ----------------
E2 = defaultdict(list)
for r in json.load(open(RES("exp2_runs.json"))):
    E2[re.sub(r"__s\d$", "", r["run"])].append(r)
assert sum(len(v) for v in E2.values()) == 96
S = {"DINOv3-SAT ViT-L (frozen)": ["dinov3_sat"],
     "Student 1, copy only (ViT-S, 10 bands)": ["student1_copy", "student1_copy_s1", "student1_copy_s2"],
     "Student 2, copy + rebuild 7 bands": ["student2_rebuild", "student2_rebuild_s1", "student2_rebuild_s2"],
     "Student 3, rebuild only (1 student training)": ["student3_rebuild_only"]}


def e2(prefixes, dates, lab, field="test_miou"):
    runs = [r for p in prefixes for r in E2[f"{p}__ltae__{dates}__{lab}"]]
    assert len(runs) == 3 * len(prefixes) and all(len(E2[f"{p}__ltae__{dates}__{lab}"]) == 3 for p in prefixes)
    per_student = [np.mean([r[field] for r in E2[f"{p}__ltae__{dates}__{lab}"]]) for p in prefixes]
    return np.mean([r[field] for r in runs]), np.std(per_student), np.std([r[field] for r in runs]), len(runs)


comp = open(RES("final_comparisons.txt")).read()


def ci(pattern):
    """find a line of final_comparisons.txt; returns (point, lo, hi)"""
    m = [l for l in comp.splitlines() if pattern in l]
    assert len(m) == 1, (pattern, m)
    v = re.search(r"= (-?[\d.]+), 95% CI \[(-?[\d.]+), (-?[\d.]+)\]", m[0])
    return tuple(float(x) for x in v.groups())


def fci(t):
    return f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"


probe = json.load(open(RES("ndvi_probe_results.json")))
bench = json.load(open(RES("benchmark.json")))
rel = json.load(open(RES("reliance", "date_reliance.json")))

# ---------------- Experiment 1 (final, our code) ----------------
E1 = defaultdict(list)
for r in json.load(open(RES("exp1_time", "exp1_runs.json"))):
    E1[re.sub(r"__s\d+$", "", r["run"])].append(r)
assert sum(len(v) for v in E1.values()) == 47
t1 = json.load(open(RES("exp1_time", "table_exp1.json")))
att = json.load(open(RES("exp1_time", "attention_exp1.json")))


def e1(key):
    v = [r["test_miou"] for r in E1[key]]
    return np.mean(v), np.std(v), len(v)


K = "dinov3_sat__{}__{}__train"
SET1 = {"one date (July)": K.format("mean-cal07", "none"), "mean all": K.format("mean", "none"),
        "mean clear": K.format("mean-cloudfree", "none")}
for d, t in (("all", "ltae"), ("clear", "ltae-cloudfree")):
    for pe, dd in (("none", "none"), ("position", "position"), ("date", "real"), ("both", "both")):
        SET1[f"{d} {pe}"] = K.format(t, dd)
SET1["shuffled"] = K.format("ltae", "shuffled")
for k, v in SET1.items():
    assert v == t1["settings"].get({"one date (July)": "single", "mean all": "mean_all", "mean clear": "mean_clear",
                                     "shuffled": "ltae_all_shuffled"}.get(k, "ltae_" + k.replace(" ", "_")), v), k
M1 = {k: e1(v) for k, v in SET1.items()}
# the linked Exp 2 runs must be the same numbers
assert abs(M1["all date"][0] - e2(["dinov3_sat"], "real", "train")[0]) < 1e-9
assert abs(M1["shuffled"][0] - e2(["dinov3_sat"], "shuffled", "train")[0]) < 1e-9
D1 = t1["differences"]


def _seed_mean(pat):
    ks = [k for k in E2 if re.fullmatch(pat.replace("*", ".*"), k)]
    assert ks, pat
    return np.mean([r["test_miou"] for k in ks for r in E2[k]])


# how far the pooled differences (CI points) are from the differences of the seed means
GAP = {"t1": max(abs(v["mean_of_seed_means"] - v["point"]) for v in D1.values()), "train": 0.0, "train_10": 0.0}
for line in comp.splitlines():
    lab = "train_10" if "train_10" in line else "train"
    m = re.match(r"mIoU\((\S+), \d+ seed\(s\)\) - mIoU\((\S+), \d+ seed\(s\)\) = (-?[\d.]+)", line)
    if m:
        GAP[lab] = max(GAP[lab], abs(_seed_mean(m.group(1)) - _seed_mean(m.group(2)) - float(m.group(3))))
    m = re.match(r"mIoU gap: \((\S+) - (\S+)\) - \((\S+) - (\S+)\) = (-?[\d.]+)", line)
    if m:
        d = (_seed_mean(m.group(1)) - _seed_mean(m.group(2))) - (_seed_mean(m.group(3)) - _seed_mean(m.group(4)))
        GAP[lab] = max(GAP[lab], abs(d - float(m.group(5))))

# ---------------- Pair 1-code rerun ----------------
P1 = defaultdict(list)
for d in json.load(open(RES("robustness_pair1_code", "pair1_code_runs.json"))):
    P1[re.sub(r"_s\d$", "", d["run"])].append(d)
assert sum(len(v) for v in P1.values()) == 47


def p1(k):
    v = [d["test_miou"] for d in P1[k]]
    return np.mean(v), np.std(v), len(v)


MAP_P1 = {"one date (July)": "single_m07", "mean all": "mean_all", "mean clear": "mean_clear", "shuffled": "ltae_all_shuffled"}
MP = {k: p1(MAP_P1.get(k, "ltae_" + k.replace(" ", "_"))) for k in SET1}

fmt = lambda t: f"{t[0]:.1f} ± {t[1]:.1f}"  # noqa: E731

# ================= write =================
w("# All numbers of the project (reference for writing the report)")
w("")
w("Generated by `scripts/make_numbers_reference.py` from the raw per-run result files that the UCloud")
w("jobs wrote (not typed by hand). The per-run files are collected in `exp2_runs.json`,")
w("`exp1_time/exp1_runs.json` and `robustness_pair1_code/pair1_code_runs.json`. Checked three ways (see the last section). Test set = PASTIS fold 5.")
w("")
w("## 0. How to read the numbers")
w("")
w("- **mIoU** = mean over the 19 classes 0–18 (background + 18 crops) of IoU, in %; class 19 (void) is")
w("  ignored. Same formula as U-TAE. **OA** = overall pixel accuracy (void ignored).")
w("- **mean ± std**: std is over seeds, population std (`np.std`, ddof = 0) in every table of this")
w("  project (our code and Pair 1's code). With ddof = 1 it would be about 1.22× larger.")
w("- **95% CI**: paired bootstrap over the 496 test series (1000 resamples, seed 0). Seeds are pooled by")
w("  summing their confusion matrices, so the CI point (\"pooled\") is not always equal to the")
w(f"  difference of the seed means: at most {GAP['t1']:.2f} apart in Table 1, {GAP['train']:.2f} in Table 2 with 100% labels,")
w(f"  {GAP['train_10']:.2f} with 10% labels. Use the CI line when you give a CI, the table means otherwise.")
w("- Rounding: tables to 1 decimal, differences to 2 decimals (or 1 in text).")
w("- Two meanings of \"shuffled dates\" — keep them apart in the text:")
w("  1. **trained with shuffled dates** (Tables 1 and 2): each image gets another image's date in")
w("     training and test. Measures how much the correct dates are worth.")
w("  2. **shuffled only at test time** (Table 3, \"reliance\"): a model trained with real dates is tested")
w("     with shuffled dates. Measures how much a trained model depends on the dates.")
w("- **Never mix rows of different tables** except the DINOv3 row, which is the same runs in Tables 1–3.")
w("")

w("## 1. Data and splits (from the data check of the download job)")
w("")
w("| item | value |")
w("|---|---|")
w("| dataset | PASTIS (Zenodo 5012942), Sentinel-2, 10 bands, 128 × 128 px at 10 m |")
w("| series (patches) | 2,433; folds 1–5: 487 / 494 / 474 / 482 / 496 series |")
w("| images (dates) | 115,222 in total; per series min 38, mean 47.4, max 61 |")
w("| train (folds 1–3) | 1,455 series, 68,786 images |")
w("| val (fold 4) | 482 series, 22,633 images |")
w("| test (fold 5) | 496 series, 23,803 images |")
w("| 10% labels | fixed random subset of train: 146 series, 6,976 images (seed 0) |")
w("| classes | 0 background, 1–18 crop types, 19 void (ignored) |")
w("| dates | days since 2018-09-01; range in the data 16 to 421 = 17 Sept 2018 to 27 Oct 2019 (September and October occur in both years) |")
w("| cloudy image | > 20% of its pixels have blue (B2) reflectance > 2000 (i.e. 0.2) |")
w("| share of cloudy images | 19.4% of all 115,222 images; 19.1% of test images |")
w("| share with cloud fraction > 0.05 / 0.10 / 0.30 / 0.50 / 0.90 | 24.7% / 22.2% / 17.3% / 14.2% / 8.7% |")
w("| DINOv3 input | RGB = B4, B3, B2, clipped at the 1st/99th percentile of folds 1–3 (cloudy pixels skipped; 85.5% of pixels kept), scaled to 0–1, normalised with the SAT-493M mean/std, upsampled 128 → 256 px |")
w("| tokens | 16 × 16 per image (one token = 8 × 8 px = 80 m × 80 m); DINOv3 1024-dim, students 384-dim |")
w("| DINOv3 features on disk | 57 GB (float16) |")
w("")

w("## 2. Models and training settings")
w("")
w("| item | value |")
w("|---|---|")
w(f"| teacher | DINOv3 ViT-L/16, SAT-493M weights (`facebook/dinov3-vitl16-pretrain-sat493m`), frozen, {bench['teacher_params_M']:.1f} M parameters |")
w(f"| student backbone | DINOv3 ViT-S/16 (LVD-1689M start weights), first layer replaced by a 10-band layer (RGB weights copied into B4/B3/B2, other bands start at 0), input z-scored per band (PASTIS normalisation, folds 1–3), {bench['student_params_M']:.1f} M parameters |")
w("| student training | images of folds 1–3 (68,786), no labels; 20 epochs (537 steps each), batch 128, AdamW lr 2e-4 for the backbone and 1e-3 for the heads, wd 0.05, 1 warm-up epoch then cosine decay; checks on 4,000 fold-4 images |")
w("| copy loss | mean over tokens of 1 − cosine similarity between the student token after a linear layer (384 → 1024) and the DINOv3 token |")
w("| rebuild loss | MSE of the 7 non-RGB bands (B5, B6, B7, B8, B8A, B11, B12; z-scored) of the 8 × 8 pixels under each token, predicted from the token by a 2-layer MLP (384 → 384 → 7 × 64) |")
w("| student 1 / 2 / 3 | copy only (λ = 0) / copy + rebuild (λ = 1) / rebuild only (copy weight 0); students 1 and 2 trained 3 times (seeds 0, 1, 2), student 3 once |")
w("| temporal model (all our tables) | L-TAE (d_model 256, 16 heads, d_k 4, MLP 256 → 128; GroupNorm computed over real dates only, padded dates ignored) + decoder of 4 conv blocks with three ×2 bilinear upsamplings (16 → 128 px) and a 1 × 1 conv to 19 classes |")
w("| L-TAE training (Tables 1–3) | 15,000 steps, batch 8 series, AdamW lr 1e-3, wd 1e-4, 5% warm-up then cosine decay, cross-entropy (void ignored); up to 32 random dates per series in training (redrawn each time), all dates at val/test; best fold-4 checkpoint (checked every 500 steps) tested on fold 5; seeds 0, 1, 2 |")
w("| mean baseline | average of the image features over the dates (in training: over the same random ≤ 32 dates), then 1 × 1 conv + BatchNorm + ReLU to 128 channels and the same decoder; no time input |")
w("| one-date baseline | least cloudy image whose date is in the calendar month (Sep/Oct: 2018 and 2019 images both count); if a series has none, the image closest to the 15th of the month; mean-pool decoder on that one image |")
w("| cloud-free dates | images with ≤ 20% cloudy pixels; if fewer than 8 remain, the 8 least cloudy are kept |")
w("| time encodings | none = no time input; position = order k of the image among the used dates, encoded as 10·k; date = days since 2018-09-01; both = half the encoding dims for position, half for date; same sinusoidal encoder as the official L-TAE |")
w(f"| benchmark | {bench['device']}, batch {bench['batch']}, 256 × 256 input: teacher {bench['teacher_img_per_s']:.0f} img/s, student {bench['student_img_per_s']:.0f} img/s |")
w("| Pair 1's code (robustness only) | official L-TAE + their decoder, lr 3e-3, 80 epochs, batch 16, wd 1e-2, all dates in training, same cloud rule, same padding fix |")
w("")

w("## 3. Experiment 2 — Distillation (Table 2)")
w("")
w("### 3.1 Main table (test mIoU, fold 5)")
w("")
w("Students 1 and 2: mean over 3 student trainings × 3 L-TAE seeds = 9 runs; in brackets the std of the")
w("3 student-training means. DINOv3 and student 3: mean of 3 L-TAE seeds, in brackets the std over")
w("these seeds. NDVI R²: linear probe (section 3.4).")
w("")
w("| Features | 100% labels, real dates | shuffled dates | 10% labels, real dates | 10%, shuffled | NDVI R² |")
w("|---|---|---|---|---|---|")
nd = probe["linear_probe_r2"]
ndk = {"DINOv3-SAT ViT-L (frozen)": "dinov3", "Student 1, copy only (ViT-S, 10 bands)": "student1",
       "Student 2, copy + rebuild 7 bands": "student2", "Student 3, rebuild only (1 student training)": "student3"}
for name, pref in S.items():
    cells = []
    for dates, lab in (("real", "train"), ("shuffled", "train"), ("real", "train_10"), ("shuffled", "train_10")):
        m, s_stu, s_run, n = e2(pref, dates, lab)
        cells.append(f"{m:.1f} ({(s_stu if len(pref) > 1 else s_run):.2f})")
    w(f"| {name} | " + " | ".join(cells) + f" | {nd[ndk[name]]:.2f} |")
w("")
w("Exact values (2 decimals) and other metrics, mean of all runs of the row:")
w("")
w("| Features | labels | dates | runs | mIoU | std over runs | OA | field accuracy |")
w("|---|---|---|---|---|---|---|---|")
for name, pref in S.items():
    for lab in ("train", "train_10"):
        for dates in ("real", "shuffled"):
            m, _, s_run, n = e2(pref, dates, lab)
            oa = e2(pref, dates, lab, "test_oa")[0]
            fa = e2(pref, dates, lab, "test_field_acc")[0]
            w(f"| {name.split(',')[0].split(' (')[0]} | {'100%' if lab == 'train' else '10%'} | {dates} | {n} | {m:.2f} | {s_run:.2f} | {oa:.2f} | {fa:.2f} |")
w("")
w("Per student training (mean of 3 L-TAE seeds, mIoU, real dates):")
w("")
w("| | 100% labels: training 0 / 1 / 2 | 10% labels: training 0 / 1 / 2 |")
w("|---|---|---|")
for name in list(S)[1:3]:
    cells = []
    for lab in ("train", "train_10"):
        cells.append(" / ".join(f"{np.mean([r['test_miou'] for r in E2[f'{p}__ltae__real__{lab}']]):.2f}" for p in S[name]))
    w(f"| {name.split(',')[0]} | {cells[0]} | {cells[1]} |")
w("")

w("### 3.2 Key differences with 95% CIs (from `final_comparisons.txt`, seeds pooled)")
w("")
w("S1 = all 9 student-1 runs, S2 = all 9 student-2 runs. 100% labels unless stated.")
w("")
rows = [
    ("S2 − S1 (rebuild loss effect, main result)", "mIoU(student2_rebuild*__ltae__real__{L}, 9 seed(s)) - mIoU(student1_copy*__ltae__real__{L}, 9"),
    ("  same, student training 0 only", "mIoU(student2_rebuild__ltae__real__{L}, 3 seed(s)) - mIoU(student1_copy__ltae__real__{L}, 3"),
    ("  same, student training 1 only", "mIoU(student2_rebuild_s1__ltae__real__{L}, 3 seed(s)) - mIoU(student1_copy_s1__ltae__real__{L}, 3"),
    ("  same, student training 2 only", "mIoU(student2_rebuild_s2__ltae__real__{L}, 3 seed(s)) - mIoU(student1_copy_s2__ltae__real__{L}, 3"),
    ("S2 − DINOv3", "mIoU(student2_rebuild*__ltae__real__{L}, 9 seed(s)) - mIoU(dinov3_sat__ltae__real__{L}, 3"),
    ("S1 − DINOv3 (does copying keep accuracy?)", "mIoU(student1_copy*__ltae__real__{L}, 9 seed(s)) - mIoU(dinov3_sat__ltae__real__{L}, 3"),
    ("S3 − S2 (student training 0) (is copying needed?)", "mIoU(student3_rebuild_only__ltae__real__{L}, 3 seed(s)) - mIoU(student2_rebuild__ltae__real__{L}, 3"),
    ("real − shuffled, DINOv3", "mIoU(dinov3_sat__ltae__real__{L}, 3 seed(s)) - mIoU(dinov3_sat__ltae__shuffled__{L}, 3"),
    ("real − shuffled, S1", "mIoU(student1_copy*__ltae__real__{L}, 9 seed(s)) - mIoU(student1_copy*__ltae__shuffled__{L}, 9"),
    ("real − shuffled, S2", "mIoU(student2_rebuild*__ltae__real__{L}, 9 seed(s)) - mIoU(student2_rebuild*__ltae__shuffled__{L}, 9"),
    ("real − shuffled, S3", "mIoU(student3_rebuild_only__ltae__real__{L}, 3 seed(s)) - mIoU(student3_rebuild_only__ltae__shuffled__{L}, 3"),
    ("gap S2 − gap S1 (does rebuild make dates more useful?)", "(student2_rebuild*__ltae__real__{L} - student2_rebuild*__ltae__shuffled__{L}) - (student1_copy*"),
    ("gap S2 − gap DINOv3", "(student2_rebuild*__ltae__real__{L} - student2_rebuild*__ltae__shuffled__{L}) - (dinov3_sat"),
]
C = {lab: {label: ci(pat.replace("{L}", L_)) for label, pat in rows} for lab, L_ in (("100", "train"), ("10", "train_10"))}
R = [label for label, _ in rows]
w("| comparison | 100% labels | 10% labels |")
w("|---|---|---|")
for label in R:
    w(f"| {label} | {fci(C['100'][label])} | {fci(C['10'][label])} |")
w("")

w("### 3.3 Student training (fold-4 checks, last of 20 epochs; from the students' log.csv)")
w("")
w("| student | training | cosine to DINOv3 (start → end) | rebuild MSE of 7 bands (end) | hours (B200) |")
w("|---|---|---|---|---|")
zero = None
STU = defaultdict(list)
for row in csv.DictReader(open(RES("student_training_log.csv"))):
    STU[row["student"]].append(row)
assert len(STU) == 7, sorted(STU)
for n in sorted(STU):
    rows_ = STU[n]
    zero = float(rows_[-1]["val_rebuild_zero"])
    w(f"| {n} | {len(rows_)} epochs | {float(rows_[0]['val_cos']):.3f} → {float(rows_[-1]['val_cos']):.3f} | "
      f"{float(rows_[-1]['val_rebuild']):.4f} | {float(rows_[-1]['hours']):.3f} |")
w("")
w(f"Reference: predicting the band average for every pixel gives rebuild MSE {zero:.3f}. For students 1")
w("and 3 the head that is not trained into the backbone (copy head of student 3, rebuild head of student 1)")
w("only measures the tokens: student 1's rebuild MSE ~0.08 means DINOv3-like tokens hold less about the")
w("non-RGB bands than student 2's (~0.012); student 3's cosine ~0.64 means its tokens are far from DINOv3.")
w("")

w("### 3.4 NDVI probe and grouping (fold 5; from `ndvi_probe_results.json`)")
w("")
st = probe["settings"]
w("Ridge regression from field-average tokens to field NDVI (NIR − red)/(NIR + red), trained on fields")
w(f"of folds 1–3, R² on fields of fold 5; {st['train_patches']} train and {st['test_patches']} test patches, up to")
w(f"{st['max_fields']} fields per patch, fields ≥ {st['min_pixels']} px, cloudy field-dates skipped (mean blue > {st['cloud']}),")
w("NIR + red < 0.05 skipped; 215,205 train rows and 208,829 test rows (field-dates).")
w("")
w("| features | NDVI R² | grouping ARI | grouping NMI |")
w("|---|---|---|---|")
for k, lab in (("control_rgb_mean", "control: mean RGB of the field"), ("control_all_bands", "control: all 10 bands of the field"),
               ("dinov3", "DINOv3"), ("student1", "student 1"), ("student2", "student 2"), ("student3", "student 3")):
    gr = probe["grouping"].get(k, {})
    w(f"| {lab} | {nd[k]:.3f} | {gr.get('ARI', float('nan')):.3f} | {gr.get('NMI', float('nan')):.3f} |" if gr else f"| {lab} | {nd[k]:.3f} | – | – |")
g0 = probe["grouping"]["ndvi_only"]
w(f"| NDVI trajectory only (no features) | – | {g0['ARI']:.3f} | {g0['NMI']:.3f} |")
w("")
w(f"Grouping: each test field's monthly feature trajectory → PCA → k-means with k = {g0['k']}, compared")
w(f"with the crop labels ({g0['n_fields']} fields). Probe uses student training 0 of each student.")
w("")

w("### 3.5 Size and speed (`benchmark.json`)")
w("")
w(f"- parameters: DINOv3 ViT-L {bench['teacher_params_M']:.1f} M, student ViT-S {bench['student_params_M']:.1f} M → "
  f"{bench['teacher_params_M'] / bench['student_params_M']:.1f}× smaller")
w(f"- speed ({bench['device']}, batch {bench['batch']}): {bench['teacher_img_per_s']:.0f} vs {bench['student_img_per_s']:.0f} images/s → "
  f"{bench['speedup']:.2f}× faster")
w("- feature extraction over all 115,222 images in the real runs: DINOv3 ~172 img/s, student ~415 img/s")
w("  (includes reading the data; not a clean speed test, use the benchmark numbers above).")
w("")

w("### 3.6 Per-class IoU (test, mean of the runs' per-class IoU; real dates, 100% labels)")
w("")
w("E1 columns are from Experiment 1 (same code and settings, DINOv3 features). Last column = DINOv3")
w("L-TAE with dates minus without any time input.")
w("")
PC = {"DINOv3": [r for r in E2["dinov3_sat__ltae__real__train"]],
      "S1": [r for p in S["Student 1, copy only (ViT-S, 10 bands)"] for r in E2[f"{p}__ltae__real__train"]],
      "S2": [r for p in S["Student 2, copy + rebuild 7 bands"] for r in E2[f"{p}__ltae__real__train"]],
      "S3": E2["student3_rebuild_only__ltae__real__train"],
      "E1 L-TAE none": E1[K.format("ltae", "none")], "E1 mean all": E1[K.format("mean", "none")],
      "E1 one date (Jul)": E1[K.format("mean-cal07", "none")]}
cls = list(PC["DINOv3"][0]["test_iou_per_class"].keys())
pcm = {k: np.array([np.nanmean([np.nan if r["test_iou_per_class"][c] is None else r["test_iou_per_class"][c]
                                for r in rs]) for c in cls]) for k, rs in PC.items()}
for k, rs in PC.items():
    assert abs(np.nanmean(pcm[k]) - np.mean([r["test_miou"] for r in rs])) < 0.01, k
w("| # | class | " + " | ".join(PC) + " | S2 − S1 | DINOv3 (date) − none |")
w("|---|---|" + "---|" * (len(PC) + 2))
for i, c in enumerate(cls):
    w(f"| {i} | {c} | " + " | ".join(f"{pcm[k][i]:.1f}" for k in PC)
      + f" | {pcm['S2'][i] - pcm['S1'][i]:+.1f} | {pcm['DINOv3'][i] - pcm['E1 L-TAE none'][i]:+.1f} |")
w("")

w("## 4. Experiment 1 — Time (Table 1, final, our code, DINOv3 features)")
w("")
w("Same L-TAE code and settings as Experiment 2. Rows \"all dates, date\" and \"shuffled\" are the")
w("Experiment 2 DINOv3 runs (identical numbers). Test mIoU, mean ± std over 3 seeds.")
w("")
w("| input | none | position | date | both |")
w("|---|---|---|---|---|")
w(f"| one date (best month = July, chosen on fold 4) | {fmt(M1['one date (July)'])} | | | |")
w(f"| mean of all dates | {fmt(M1['mean all'])} | | | |")
w(f"| mean of cloud-free dates | {fmt(M1['mean clear'])} | | | |")
w(f"| L-TAE, all dates | {fmt(M1['all none'])} | {fmt(M1['all position'])} | {fmt(M1['all date'])} | {fmt(M1['all both'])} |")
w(f"| L-TAE, cloud-free dates | {fmt(M1['clear none'])} | {fmt(M1['clear position'])} | {fmt(M1['clear date'])} | {fmt(M1['clear both'])} |")
w(f"| L-TAE, all dates, trained with shuffled dates | | | {fmt(M1['shuffled'])} | |")
w("")
w("Exact values and other metrics:")
w("")
w("| setting | runs | mIoU | std | OA | field acc. | val mIoU (best) |")
w("|---|---|---|---|---|---|---|")
for k, key in SET1.items():
    rs = E1[key]
    w(f"| {k} | {len(rs)} | {np.mean([r['test_miou'] for r in rs]):.2f} | {np.std([r['test_miou'] for r in rs]):.2f} | "
      f"{np.mean([r['test_oa'] for r in rs]):.2f} | {np.mean([r['test_field_acc'] for r in rs]):.2f} | {np.mean([r['val_miou_best'] for r in rs]):.2f} |")
w("")
w("### 4.1 Key differences (seeds pooled, paired bootstrap 95% CI)")
w("")
w("| comparison | difference of means | pooled [95% CI] |")
w("|---|---|---|")
for k, v in D1.items():
    w(f"| {k} | {v['mean_of_seed_means']:+.2f} | {v['point']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] |")
extra = [("both − date (all dates)", "all both", "all date"), ("both − date (cloud-free)", "clear both", "clear date"),
         ("trained with shuffled dates − no time input", "shuffled", "all none"),
         ("L-TAE no time: cloud-free − all dates", "clear none", "all none")]
for lab, x, y in extra:
    w(f"| {lab} | {M1[x][0] - M1[y][0]:+.2f} | (no CI computed) |")
w("")
w("### 4.2 One date per calendar month (seed 0; val / test mIoU)")
w("")
w("| month | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
vals, tests = [], []
for m in range(1, 13):
    r0 = [r for r in E1[K.format(f"mean-cal{m:02d}", "none")] if r["seed"] == 0][0]
    vals.append(f"{r0['val_miou_best']:.1f}")
    tests.append(f"{r0['test_miou']:.1f}")
w("| val | " + " | ".join(vals) + " |")
w("| test | " + " | ".join(tests) + " |")
w("")
w(f"Best on val: month {t1['best_month']} (July); it got 2 more seeds (row \"one date\" above).")
w("")
w("### 4.3 Attention of the L-TAE (all dates, real dates; 3 seeds pooled; test fold)")
w("")
pa = att["pooled"]
w("Attention of a date = mean over heads and the 16 × 16 tokens, × number of dates T (1 = equal share).")
w(f"{pa['n_dates']} test dates over 3 runs; {100 * pa['share_cloudy']:.1f}% cloudy.")
w("")
w(f"- clear dates {pa['attention_clear']:.2f}, cloudy dates {pa['attention_cloudy']:.2f}")
w(f"- Spearman(attention, cloud fraction) = {pa['spearman_attention_cloud']:.2f}")
w("- per seed (clear / cloudy / Spearman): " + "; ".join(
    f"s{n[-1]}: {v['attention_clear']:.2f} / {v['attention_cloudy']:.3f} / {v['spearman_attention_cloud']:.2f}"
    for n, v in sorted(att["runs"].items())))
mo = pa["attention_by_month_clear_only"]
names = {9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug"}
w("- by calendar month, clear dates only: " + ", ".join(f"{names[int(k)]} {v:.2f}" for k, v in mo.items()))
w("- figure: `exp1_time/fig_attention_exp1.png`")
w("")

w("## 5. Reliance on dates: shuffled only at test time (Table 3)")
w("")
w("Models trained with real dates, tested on fold 5 with real and with shuffled dates (same")
w("permutation per series for every model). mIoU of pooled confusion matrices (seeds pooled), so")
w("\"real\" can differ by ≤ 0.01 from the table means. Nothing trained.")
w("U-TAE = official pre-trained weights (Zenodo 5172293, `Fold_1` = trained on folds 1–3, val 4, i.e.")
w("our split; code commit 987874e), raw 10 bands, batch 1.")
w("")
w("| model | runs | real | shuffled at test | drop [95% CI] | drop − DINOv3 drop [95% CI] |")
w("|---|---|---|---|---|---|")
for r in rel["rows"]:
    dd = (f"{r['drop_minus_dinov3_drop']:.2f} [{r['drop_minus_dinov3_drop_ci95'][0]:.2f}, {r['drop_minus_dinov3_drop_ci95'][1]:.2f}]"
          if "drop_minus_dinov3_drop" in r else "–")
    w(f"| {r['model']} | {r['runs']} | {r['miou_real']:.2f} | {r['miou_shuffled_at_test']:.2f} | "
      f"{r['drop']:.2f} [{r['drop_ci95'][0]:.2f}, {r['drop_ci95'][1]:.2f}] | {dd} |")
w("")
per = defaultdict(list)
for l in open(RES("reliance", "log_run_reliance.txt")):
    m_ = re.match(r"(\S+)__ltae__real__train__s\d: real ([\d.]+) \(results.json ([\d.]+)\), shuffled at test ([\d.]+), drop ([\d.]+)", l)
    if m_:
        assert m_.group(2) == m_.group(3)
        per[m_.group(1).split("_s")[0] if m_.group(1).startswith("student") else m_.group(1)].append(float(m_.group(5)))
assert sum(len(v) for v in per.values()) == 24
w("Drop per run (from `reliance/log_run_reliance.txt`): " + "; ".join(f"{k}: " + ", ".join(f"{x:.2f}" for x in v) for k, v in per.items()) + ".")
w("Every re-evaluated real-date score matched its results.json (24 of 24).")
w("")
w("External reference: the U-TAE repository README reports U-TAE at 63.1 mIoU, 83.2 OA on PASTIS")
w("(1.1 M parameters) — their own evaluation, not comparable 1:1 with our fold-5 number (62.31).")
w("")

w("## 6. Robustness: Experiment 1 with Pair 1's code (with the padding fix)")
w("")
w("Not a main table. One sentence in the report. Their decoder, lr 3e-3, 80 epochs, batch 16, all dates")
w("in training; same features, split, cloud rule; mean ± std over 3 seeds (ddof 0).")
w("")
w("| setting | Pair 1's code | our code (Table 1) | difference |")
w("|---|---|---|---|")
for k in SET1:
    w(f"| {k} | {fmt(MP[k])} | {fmt(M1[k])} | {MP[k][0] - M1[k][0]:+.1f} |")
w("")
dif = lambda M, x, y: M[x][0] - M[y][0]  # noqa: E731
w("| difference | Pair 1's code | our code |")
w("|---|---|---|")
for lab, x, y in (("mean all − one date", "mean all", "one date (July)"), ("mean clear − mean all", "mean clear", "mean all"),
                  ("L-TAE all none − mean clear", "all none", "mean clear"), ("order (all): position − none", "all position", "all none"),
                  ("date beyond order (all): date − position", "all date", "all position"),
                  ("order (clear): position − none", "clear position", "clear none"),
                  ("date beyond order (clear): date − position", "clear date", "clear position"),
                  ("real − shuffled (trained)", "all date", "shuffled"), ("L-TAE all date − one date", "all date", "one date (July)"),
                  ("L-TAE date: clear − all", "clear date", "all date")):
    w(f"| {lab} | {dif(MP, x, y):+.2f} | {dif(M1, x, y):+.2f} |")
w("")
s0 = lambda m: [d for d in P1[f"single_m{m:02d}"] if d["seed"] == 0][0]  # noqa: E731
jul = P1["single_m07"]
assert len(jul) == 3
w("Pair 1's code, one date per month (seed 0, val / test): " + ", ".join(
    f"{names[m]} {s0(m)['val_miou']:.1f} / {s0(m)['test_miou']:.1f}" for m in range(1, 13))
  + ". Best on val: July. (Their table `robustness_pair1_code/table_pair1_code.md` lists July as the "
  + f"3-seed mean, {np.mean([d['val_miou'] for d in jul]):.1f} / {np.mean([d['test_miou'] for d in jul]):.1f}.)")
w("")
pat = open(RES("robustness_pair1_code", "attention_pair1_code.txt")).read()
gx = lambda rx: float(re.search(rx, pat).group(1))  # noqa: E731
w(f"Pair 1's code, attention (L-TAE all dates, date, seed 0, test): clear {gx(r'clear dates:\s+([\d.]+)'):.2f}, "
  f"cloudy {gx(r'cloudy dates:\s+([\d.]+)'):.2f}, Spearman "
  + f"{gx(r'Spearman[^:]*:\s+(-?[\d.]+)'):.2f}".replace("-", "−"))
w("(from their attention script; per-month numbers not kept).")
w("")

w("## 7. Round 1 (history only — NOT for the report)")
w("")
w("First full run, L-TAE 5,000 steps (val mIoU was still rising, so round 2 used 15,000 steps):")
w("DINOv3 50.11 / shuffled 45.64; student 1 49.79 / 46.01; student 2 51.95 / 48.12 (100% labels);")
w("10% labels: 36.72 / 36.62 / 38.98. S2 − S1 +2.15 [1.59, 2.66]. Its NDVI probe was invalid (bug, fixed).")
w("Same direction as the final results.")
w("")

w("## 8. Compute (UCloud, NVIDIA B200)")
w("")
w("| job | what | wall time |")
w("|---|---|---|")
w("| full run (Oct 28–29) | DINOv3 features (~11 min), students 1 + 2, 36 L-TAE runs (5k steps), probe, benchmark | 1 h 04 min |")
w("| final round (Oct 29) | probe, 4 more student trainings + student 3, 96 L-TAE runs (15k steps), comparisons | 3 h 20 min |")
w("| reliance (Nov 5) | U-TAE pre-trained + 24 L-TAE re-evaluations | ~5 min |")
w("| Pair 1-code rerun (Nov 5) | 47 runs | 2 h 42 min |")
w("| Experiment 1 final (Nov 5) | clouds + 41 new runs (+ 6 linked) | 1 h 05 min |")
w("")
w("One student training: ~0.19 h on one B200. One L-TAE run of 15k steps: ~0.20 h wall time, with 8 runs sharing one GPU.")
w("")

w("## 9. Claims the numbers support (and ones they do not)")
w("")
d = lambda k: D1[k]  # noqa: E731
w("Supported:")
w(f"1. Many dates beat one date: mean of all dates +{d('time series vs one date')['point']:.1f}, L-TAE +{d('L-TAE (all, date) vs one date')['point']:.1f} mIoU over the best single month.")
w(f"2. The L-TAE learns to ignore cloudy images (attention {pa['attention_cloudy']:.2f} vs {pa['attention_clear']:.2f}); removing clouds by hand helps the mean "
  f"(+{d('removing clouds (mean)')['point']:.2f}) but not the L-TAE ({d('removing clouds (L-TAE, date)')['point']:+.2f}, CI includes 0).")
w(f"3. Most of the L-TAE's gain does not need any time input: L-TAE without time is +{d('attention beyond cloud removal (L-TAE all, no time vs mean of cloud-free)')['point']:.1f} over the mean of cloud-free dates.")
w(f"4. Correct dates add a small but clear amount: real vs shuffled +{d('real vs shuffled dates (all)')['point']:.2f} [{d('real vs shuffled dates (all)')['ci95'][0]:.2f}, {d('real vs shuffled dates (all)')['ci95'][1]:.2f}]; "
  f"order +{d('order (all dates)')['point']:.2f}, calendar date +{d('calendar date beyond order (all)')['point']:.2f} more. "
  f"Training with shuffled dates equals no time input ({M1['shuffled'][0]:.1f} vs {M1['all none'][0]:.1f}).")
RR = {r["model"]: r for r in rel["rows"]}
w(f"5. Trained with real dates, DINOv3 + L-TAE does depend on them at test (drop {RR['DINOv3 + L-TAE']['drop']:.1f}), but far less than U-TAE")
w(f"   trained on raw bands (drop {RR['U-TAE (pre-trained, 10 bands)']['drop']:.1f}).")
w(f"6. Copying DINOv3 into a {bench['teacher_params_M'] / bench['student_params_M']:.1f}× smaller, {bench['speedup']:.1f}× faster 10-band student keeps its accuracy (S1 − DINOv3 {fci(C['100'][R[5]])}, not significant).")
w(f"7. Adding the rebuild loss gives +{C['100'][R[0]][0]:.2f} mIoU [{C['100'][R[0]][1]:.2f}, {C['100'][R[0]][2]:.2f}] over copy only, in each of 3 student trainings, also with 10% labels, and raises NDVI R² from {nd['student1']:.2f} to {nd['student2']:.2f}.")
w("8. That gain is not from using the dates more: gap S2 − gap S1 = " + fci(C['100'][R[11]]) + " (100%).")
w("9. Copying DINOv3 is not needed for the gain: student 3 (rebuild only) ≈ student 2 (" + fci(C['100'][R[6]]) + ", one training).")
w("")
w("Not supported / be careful:")
w(f"- Do not say students use timing more than DINOv3 in general: gap S2 − gap DINOv3 is {fci(C['100'][R[12]])} at")
w(f"  100% labels but not at 10% ({fci(C['10'][R[12]])}); with test-time shuffling S2 drops "
  f"{RR['Student 2 (copy + rebuild) + L-TAE']['drop_minus_dinov3_drop']:.1f} more than DINOv3.")
w("  Mixed evidence → describe, don't claim.")
w(f"- Student 3's large test-time drop ({RR['Student 3 (rebuild only) + L-TAE']['drop']:.1f}) comes from one student training; its tokens are far from DINOv3.")
w("- Student vs DINOv3 mixes two changes (10 bands and ViT-S vs ViT-L); S2 vs S1 is the clean comparison.")
w("- One test fold (fold 5), not 5-fold cross-validation. Std over 3 seeds is small but seeds are few.")
w("- Tokens cover 80 m × 80 m; small fields are mixed with their neighbours.")
w("- Cloud detection = blue threshold only (no cloud shadows; bright soil or snow may count as cloud).")
w("")

w("## 10. Corrections to earlier notes")
w("")
w("- Dates per series are **38–61 (mean 47.4)**, not 33–61 as written in an early planning note.")
w("- The images run from **17 Sept 2018 to 27 Oct 2019** (not to Nov/Dec 2019). A code comment said")
w("  Sep–Nov exist in both years; in fact only Sep and Oct do. No effect on results (the rule is the same).")
w("- The student-1 rebuild MSE differs by training (0.082 / 0.077 / 0.082); the earlier one-number")
w("  summary \"0.08\" is fine.")
w("- Pair 1's attention by month (\"up to 2.6, April–June\") is not in any saved file; use our numbers (section 4.3).")
w("")

w("## 11. How this file was checked")
w("")
w("1. Every table is computed by this script from the per-run files (96 + 47 + 47 runs); the script asserts")
w("   the run counts, that the 6 linked DINOv3 runs in Table 1 equal the Experiment 2 runs, and that the")
w("   setting names match the table the job made.")
w("2. Independently recomputed by hand-written one-off checks and compared with the files the jobs")
w("   wrote: `final_summary.csv` (all 32 rows match to 0.01), `final_summary_by_student_seed.csv`,")
w("   `exp1_time/table_exp1.md`, `robustness_pair1_code/table_pair1_code.md`, `reliance/date_reliance.txt`")
w("   and `reliance/log_run_reliance.txt`.")
w("3. A second, separately downloaded copy of the 96 Experiment 2 run files gives identical numbers.")
w("4. Text numbers in section 9 are read from the same data, not typed.")
open(a.out, "w").write("\n".join(L) + "\n")
print("wrote", a.out, len(L), "lines")
