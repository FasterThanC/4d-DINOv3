"""Size and speed of DINOv3 (teacher) vs a student: parameters and images per second.

    python scripts/benchmark.py --ckpt $RUNS/students/student2_rebuild/best.pt --out $RUNS/benchmark.json
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dino import load_backbone  # noqa: E402
from src.student import student_from_checkpoint  # noqa: E402
from src.utils import autocast, get_device, save_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", default="facebook/dinov3-vitl16-pretrain-sat493m")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="auto")
a = ap.parse_args()
device = get_device(a.device)


def speed(fn, x):
    with torch.inference_mode(), autocast(device):
        for _ in range(3):
            fn(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(a.iters):
            fn(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return a.iters * x.shape[0] / (time.time() - t0)


teacher = load_backbone(a.teacher, device)
student = student_from_checkpoint(torch.load(a.ckpt, map_location="cpu", weights_only=False)).eval().to(device)
xt = torch.randn(a.batch, 3, a.size, a.size, device=device)
xs = torch.randn(a.batch, 10, 128, 128, device=device)
res = {
    "device": str(torch.cuda.get_device_name(0) if device.type == "cuda" else device),
    "teacher_params_M": sum(p.numel() for p in teacher.parameters()) / 1e6,
    "student_params_M": sum(p.numel() for p in student.backbone.parameters()) / 1e6,
    "teacher_img_per_s": speed(lambda x: teacher(pixel_values=x), xt),
    "student_img_per_s": speed(lambda x: student.tokens(x), xs),
    "batch": a.batch,
}
res["speedup"] = res["student_img_per_s"] / res["teacher_img_per_s"]
print(res)
save_json(res, a.out)
