#!/bin/bash
# One-time setup on the cluster (from a git clone or an unpacked bundle): build the venv,
# verify any data present, check the torch build, and run the test suite.
#
# Usage: bash scripts/hpc/setup_env.sh [python-executable]   (needs Python >= 3.11)
set -euo pipefail

cd "$(dirname "$0")/../.."
py=${1:-python3}
"$py" -c 'import sys; assert sys.version_info >= (3, 11), f"need Python >= 3.11, got {sys.version}"'

export HF_HOME=${SCRATCH:-$HOME}/hf_cache   # model downloads go to scratch, not the home quota
"$py" -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .

if [ -f data/MANIFEST.sha256 ]; then   # only present when the project came from make_bundle.sh
  (cd data && sha256sum --check --quiet MANIFEST.sha256) && echo "data checksums OK"
fi
if [ -d data/jigsaw/splits ]; then
  .venv/bin/python -I scripts/hpc/verify_splits.py data/jigsaw/splits
fi
.venv/bin/python -c "import torch; print('torch', torch.__version__, '| CUDA build:', torch.version.cuda, '| GPU visible:', torch.cuda.is_available())"
.venv/bin/python -m pytest tests -q
