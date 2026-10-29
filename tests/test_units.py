"""Small checks of the parts that are easy to get wrong. Run: python tests/test_units.py"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dino import compute_rgb_percentiles, prepare_rgb  # noqa: E402
from src.metrics import bootstrap_gap_difference, confusion, scores  # noqa: E402
from src.pastis import NON_RGB_IDX, RGB_IDX  # noqa: E402
from src.temporal import LTAE  # noqa: E402

ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok &= bool(cond)


# 1. band indices: B2 B3 B4 B5 B6 B7 B8 B8A B11 B12
check("RGB = B4,B3,B2", RGB_IDX == [2, 1, 0])
check("non-RGB = B5..B12 (7 bands)", NON_RGB_IDX == [3, 4, 5, 6, 7, 8, 9])

# 2. cloud pixels do not set the 99th percentile
rng = np.random.RandomState(0)
img = rng.randint(200, 1500, size=(10, 10, 32, 32)).astype(np.int16)
img[:3] = 7000                                    # 3 of 10 dates fully cloudy (all bands bright)
pct = compute_rgb_percentiles([img])
check(f"percentiles ignore clouds (high={pct['high']})", max(pct["high"]) < 1600)

# 3. prepare_rgb: shape, range, upsampling
x = torch.from_numpy(img[3:5].astype(np.float32))
out = prepare_rgb(x, pct, (0.43, 0.411, 0.296), (0.213, 0.156, 0.143), size=64)
check("prepare_rgb shape", tuple(out.shape) == (2, 3, 64, 64))

# 4. mIoU equals the U-TAE formula (TP / (TP+FP+FN), nanmean over classes with TP+FP+FN > 0)
y = rng.randint(0, 20, size=(128, 128))
p = rng.randint(0, 19, size=(128, 128))
cm = confusion(p, y)
tp = np.diag(cm).astype(float)
with np.errstate(invalid="ignore", divide="ignore"):
    ref = np.nanmean(tp / (cm.sum(0) + cm.sum(1) - tp)) * 100
check("mIoU matches U-TAE formula", abs(scores(cm)["miou"] - ref) < 1e-9)
check("void label ignored", cm.sum() == (y != 19).sum())

# 5. bootstrap gap: point estimate equals the plain difference of differences
cms = [rng.randint(0, 50, size=(30, 19, 19)) for _ in range(4)]
g = bootstrap_gap_difference(*cms, n_boot=20)
m = [scores(c.sum(0))["miou"] for c in cms]
check("gap point estimate", abs(g["point"] - ((m[0] - m[1]) - (m[2] - m[3]))) < 1e-9)

# 6. L-TAE: without dates it ignores image order; with dates, wrong dates change the output
torch.manual_seed(0)
x = torch.randn(2, 9, 64, 4, 4)
days = torch.sort(torch.rand(2, 9) * 400, 1).values
perm = torch.randperm(9)
m0 = LTAE(64, use_dates=False).eval()
m1 = LTAE(64, use_dates=True).eval()
check("no-date L-TAE ignores order", (m0(x, days) - m0(x[:, perm], days)).abs().max() < 1e-4)
check("dated L-TAE: images+dates permuted together -> same output",
      (m1(x, days) - m1(x[:, perm], days[:, perm])).abs().max() < 1e-4)
check("dated L-TAE: shuffled dates -> different output", (m1(x, days) - m1(x, days[:, perm])).abs().max() > 1e-2)
pad = torch.zeros(2, 9, dtype=torch.bool)
pad[:, 7:] = True
check("padded dates have no effect",
      (m1(x, days, pad) - m1(torch.cat([x[:, :7], torch.randn(2, 2, 64, 4, 4)], 1), days, pad)).abs().max() < 1e-4)

# 7. student: a head trained on detached tokens does not change the backbone
try:
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    from src.student import Student, make_student_backbone
    import tempfile
    d = tempfile.mkdtemp()
    DINOv3ViTModel(DINOv3ViTConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=1,
                                   intermediate_size=64, num_register_tokens=4)).save_pretrained(d)
    st = Student(make_student_backbone(d), teacher_dim=48)
    xin = torch.randn(2, 10, 128, 128)
    _, proj, rec = st(xin, detach_rebuild=True)
    rec.pow(2).mean().backward()
    g = sum(p.grad.abs().sum().item() for p in st.backbone.parameters() if p.grad is not None)
    check("detached rebuild head: no gradient into the student", g == 0)
    st.zero_grad()
    _, proj, rec = st(xin)
    rec.pow(2).mean().backward()
    g = sum(p.grad.abs().sum().item() for p in st.backbone.parameters() if p.grad is not None)
    check("normal rebuild head: gradient reaches the student", g > 0)
    # rebuild output pixel layout: token (i, j) covers native pixels [8i:8i+8, 8j:8j+8]
    with torch.no_grad():
        st.rebuild[-1].weight.zero_()
        st.rebuild[-1].bias.zero_()
        st.rebuild[-1].bias[0] = 1.0                 # band 0, top-left pixel of each token
        _, _, rec = st(xin)
    check("rebuild layout", rec[0, 0, 0, 0] == 1 and rec[0, 0, 8, 8] == 1 and rec[0, 0, 1, 1] == 0)
except ImportError as e:
    print("skip student checks:", e)

print("ALL PASSED" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
