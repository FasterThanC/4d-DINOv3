#!/usr/bin/env bash
# Paths used by all scripts. Source it:  source ucloud/config.sh
#
# DATA_DIR must be a UCloud folder you ADDED to the job (added folders appear under /work/).
# Everything big lives there: PASTIS (~29 GB zip + ~40 GB unzipped), features (~100 GB),
# runs, the Python env and the Hugging Face cache. If the folder is not mounted, files written
# there would not be in your drive where you expect them, so we stop instead.
export CODE_DIR="${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export DATA_DIR="${DATA_DIR:-/work/4d-dino-data}"

if [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: DATA_DIR=$DATA_DIR does not exist."
  echo "Add your data folder to the UCloud job (it appears as /work/<folder name>) and run:"
  echo "    export DATA_DIR=/work/<folder name>"
  return 1 2>/dev/null || exit 1
fi

export PASTIS="${PASTIS:-$DATA_DIR/PASTIS}"
export FEATS="$DATA_DIR/features"
export RUNS="$DATA_DIR/runs"
export SPLITS="${SPLITS:-$CODE_DIR/splits.json}"
export VENV="$DATA_DIR/venv"
export HF_HOME="$DATA_DIR/hf_cache"

export TEACHER_MODEL="${TEACHER_MODEL:-facebook/dinov3-vitl16-pretrain-sat493m}"
export STUDENT_INIT="${STUDENT_INIT:-facebook/dinov3-vits16-pretrain-lvd1689m}"
export TEACHER_FEATS="${TEACHER_FEATS:-$FEATS/dinov3_sat}"

# uv (pre-installed in the UCloud Terminal app) keeps Python, packages and its cache in DATA_DIR,
# so the environment survives between jobs.
export UV_CACHE_DIR="$DATA_DIR/uv-cache"
export UV_PYTHON_INSTALL_DIR="$DATA_DIR/uv-python"
export UV_PYTHON_PREFERENCE=only-managed
export UV_LINK_MODE=copy
export PATH="$DATA_DIR/uv-bin:$PATH"

# CPUs this job may use (UCloud limits CPUs per machine type; nproc can show the whole server)
_ncpu () {
  if [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r q p < /sys/fs/cgroup/cpu.max
    if [ "$q" != "max" ]; then echo $(( q / p > 0 ? q / p : 1 )); return; fi
  fi
  nproc
}
export NCPU="${NCPU:-$(_ncpu)}"
export WORKERS="${WORKERS:-$(( NCPU > 17 ? 16 : (NCPU > 1 ? NCPU - 1 : 1) ))}"

mkdir -p "$FEATS" "$RUNS/logs"
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi
cd "$CODE_DIR" || return 1
