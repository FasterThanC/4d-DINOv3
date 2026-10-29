import csv
import json
import os
import random
import subprocess
import time

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device, enabled=True):
    """bf16 autocast on CUDA, no-op elsewhere."""
    if device.type == "cuda" and enabled:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


class CSVLogger:
    def __init__(self, path):
        self.path = path
        self.header = None
        if os.path.exists(path):
            with open(path) as f:
                first = f.readline().strip()
            self.header = first.split(",") if first else None

    def log(self, row):
        if self.header is None:
            self.header = list(row.keys())
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(self.header)
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") for k in self.header])


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def hours(self):
        return (time.time() - self.t0) / 3600
