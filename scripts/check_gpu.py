"""Check that PyTorch can really use this GPU (driver, CUDA build, bf16 kernels).

    python scripts/check_gpu.py          # exit code 1 if there is no usable GPU
    python scripts/check_gpu.py --cpu-ok # only report (used during setup on a CPU machine)
"""
import argparse
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--cpu-ok", action="store_true")
a = ap.parse_args()

print(f"torch {torch.__version__}, built for CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    print("FAIL no usable GPU. If this is a GPU machine, the NVIDIA driver is probably older than "
          "this PyTorch build needs (see nvidia-smi).")
    sys.exit(0 if a.cpu_ok else 1)

dev = torch.device("cuda")
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
mem = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"GPU: {name}, compute capability {cap[0]}.{cap[1]}, {mem:.0f} GB memory")
print("this PyTorch has kernels for:", " ".join(torch.cuda.get_arch_list()))
try:
    x = torch.randn(256, 3, 64, 64, device=dev)
    conv = torch.nn.Conv2d(3, 64, 16, stride=16).to(dev)
    layer = torch.nn.TransformerEncoderLayer(64, 4, 128, batch_first=True).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        t = conv(x).flatten(2).transpose(1, 2)
        y = layer(t).float().mean()
    y.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(y).item()
except Exception as e:  # noqa: BLE001
    print(f"FAIL GPU test computation failed: {e}")
    sys.exit(1)
print("PASS GPU works (bf16 conv + attention, forward and backward)")
