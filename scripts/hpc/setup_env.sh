#!/usr/bin/env bash
# One-off setup on the HPC LOGIN node, from the repo root:   bash scripts/hpc/setup_env.sh
set -euo pipefail
[[ -f scripts/hpc/cluster.env ]] && source scripts/hpc/cluster.env
python3 -c 'import sys; assert sys.version_info >= (3, 11), sys.version' 2>/dev/null || {
  echo "python3 is $(python3 --version 2>&1); 3.11+ needed. Add a 'module load' line to scripts/hpc/cluster.env."; exit 1; }
unset PYTHONPATH   # module-provided PYTHONPATH would shadow the venv's own pip
# torch + CUDA wheels are ~6 GB: build the venv on scratch (VENV_DIR) and link it as .venv, so home quota is not hit.
VENV_DIR="${VENV_DIR:-/scratch/$USER/venvs/understanding_lmms}"
rm -rf "$VENV_DIR"
[[ -L .venv || -e .venv ]] && rm -rf .venv
mkdir -p "$(dirname "$VENV_DIR")"
python3 -m venv "$VENV_DIR"
ln -s "$VENV_DIR" .venv
.venv/bin/python -m pip install --no-cache-dir --upgrade pip
.venv/bin/python -m pip install --no-cache-dir -r requirements-train.txt
.venv/bin/python -m pip install --no-cache-dir -e . --no-deps
[[ -d "${MLLM_TESTING_DIR:-../MLLM_Testing}" ]] || echo "WARNING: MLLM_Testing not found at ${MLLM_TESTING_DIR:-../MLLM_Testing} (clone it, set MLLM_TESTING_DIR)"
.venv/bin/python -c "import torch, transformers, peft; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)"
.venv/bin/python -m lmmviz.capture.run --experiment configs/experiments/baseline.yaml --list | tail -1
echo "venv ready; next: bash scripts/hpc/download_models.sh"
