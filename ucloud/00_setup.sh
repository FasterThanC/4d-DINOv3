#!/usr/bin/env bash
# One-time setup: Python + packages in $DATA_DIR (kept between jobs), using uv.
# uv is pre-installed in the UCloud Terminal app; if it is missing, it is installed here.
# Can run on a CPU machine (cheaper); the GPU itself is checked later by preflight.sh.
#   bash ucloud/00_setup.sh            # install
#   bash ucloud/00_setup.sh fix-torch  # GPU machine where PyTorch cannot use the GPU (old driver):
#                                      # installs PyTorch 2.9.1 built for CUDA 12.8 instead
# Then log in to Hugging Face once (DINOv3 models are gated):
#     source ucloud/config.sh && hf auth login        (paste a READ token)
set -euo pipefail
source "$(dirname "$0")/config.sh"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found, installing it into $DATA_DIR/uv-bin"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$DATA_DIR/uv-bin" UV_NO_MODIFY_PATH=1 sh
fi
export PATH="$DATA_DIR/uv-bin:$PATH"

if [ "${1:-}" = "fix-torch" ]; then
  source "$VENV/bin/activate"
  nvidia-smi --query-gpu=name,driver_version --format=csv || true
  uv pip install --reinstall "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128
  python scripts/check_gpu.py
  exit 0
fi

if [ ! -f "$VENV/bin/activate" ]; then
  uv venv "$VENV" --python 3.12
fi
source "$VENV/bin/activate"
uv pip install -r "$CODE_DIR/requirements.txt"
python -c "import torch, transformers, sklearn; print('torch', torch.__version__, '| transformers', transformers.__version__)"
python scripts/check_gpu.py --cpu-ok
echo
echo "Setup done. Next (once): source ucloud/config.sh && hf auth login"
echo "Then: bash ucloud/01_download_pastis.sh"
