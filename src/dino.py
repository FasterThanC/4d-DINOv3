"""DINOv3 (teacher) loading, RGB preparation and patch-token extraction.

The teacher is frozen. Each Sentinel-2 image (128 x 128 px, 10 m) is turned into RGB
(B4, B3, B2), clipped at fixed percentiles, scaled to 0-1, normalised with the model's
mean/std, upsampled (default 128 -> 256 px) and passed through DINOv3.
With patch size 16 this gives a 16 x 16 grid of tokens (one token per 80 x 80 m).
"""
import json

import numpy as np
import torch
import torch.nn.functional as F

from .pastis import RGB_IDX

SAT_MEAN, SAT_STD = (0.430, 0.411, 0.296), (0.213, 0.156, 0.143)        # DINOv3 SAT-493M
WEB_MEAN, WEB_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)        # web / ImageNet models


def default_mean_std(model_name, norm="auto"):
    """norm: "sat", "web" or "auto" (sat if the model name contains "sat493m").
    Values from the official DINOv3 README (github.com/facebookresearch/dinov3)."""
    if norm == "auto":
        norm = "sat" if "sat493m" in model_name.lower() else "web"
    return (SAT_MEAN, SAT_STD) if norm == "sat" else (WEB_MEAN, WEB_STD)


def load_backbone(model_name, device, dtype=torch.float32):
    """Load a DINOv3 / DINOv2 backbone from the Hugging Face hub or a local folder."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_name, dtype=dtype)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def num_prefix_tokens(model):
    """CLS token + register tokens that come before the patch tokens."""
    return 1 + int(getattr(model.config, "num_register_tokens", 0) or 0)


def patch_tokens(model, pixel_values):
    """Run the backbone and return patch tokens as (B, D, h, w)."""
    out = model(pixel_values=pixel_values)
    x = out.last_hidden_state[:, num_prefix_tokens(model):, :]
    B, N, D = x.shape
    ps = model.config.patch_size
    h, w = pixel_values.shape[-2] // ps, pixel_values.shape[-1] // ps
    assert h * w == N, f"expected {h}x{w} tokens, got {N}"
    return x.transpose(1, 2).reshape(B, D, h, w)


def compute_rgb_percentiles(arrays, low=1.0, high=99.0, max_pixels=5_000_000, seed=0,
                            cloud_blue=2000):
    """Per-band (R, G, B) percentiles from a list of (T, 10, H, W) arrays.

    Pixels with blue reflectance above `cloud_blue` (0.2 x 10000, i.e. bright clouds) are left
    out. PASTIS keeps partly cloudy images; with clouds included, the 99th percentile becomes
    the cloud brightness and all fields end up dark with little contrast.
    """
    rng = np.random.RandomState(seed)
    vals = [[], [], []]
    per_arr = max(1, max_pixels // max(1, len(arrays)))
    n_all = n_kept = 0
    for a in arrays:
        rgb = np.asarray(a[:, RGB_IDX], dtype=np.float32)          # T, 3, H, W  (R=B4, G=B3, B=B2)
        flat = rgb.transpose(1, 0, 2, 3).reshape(3, -1)
        clear = flat[2] <= cloud_blue
        n_all += flat.shape[1]
        n_kept += int(clear.sum())
        flat = flat[:, clear]
        if flat.shape[1] == 0:
            continue
        idx = rng.choice(flat.shape[1], size=min(per_arr, flat.shape[1]), replace=False)
        for c in range(3):
            vals[c].append(flat[c, idx])
    lo = [float(np.percentile(np.concatenate(v), low)) for v in vals]
    hi = [float(np.percentile(np.concatenate(v), high)) for v in vals]
    return {"low_pct": low, "high_pct": high, "low": lo, "high": hi, "bands": ["B4", "B3", "B2"],
            "cloud_blue_threshold": cloud_blue, "share_pixels_kept": n_kept / max(n_all, 1)}


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def prepare_rgb(s2, pct, mean, std, size=256):
    """(N, 10, H, W) raw reflectance tensor -> (N, 3, size, size) normalised RGB tensor."""
    rgb = s2[:, RGB_IDX].float()
    lo = torch.tensor(pct["low"], device=rgb.device).view(1, 3, 1, 1)
    hi = torch.tensor(pct["high"], device=rgb.device).view(1, 3, 1, 1)
    rgb = ((rgb - lo) / (hi - lo)).clamp(0, 1)
    m = torch.tensor(mean, device=rgb.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=rgb.device).view(1, 3, 1, 1)
    rgb = (rgb - m) / s
    if rgb.shape[-1] != size:
        rgb = F.interpolate(rgb, size=(size, size), mode="bilinear", align_corners=False)
    return rgb
