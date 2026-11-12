# Sourced by every .slurm job (not a job itself).
set -euo pipefail
[[ -f scripts/hpc/cluster.env ]] && source scripts/hpc/cluster.env
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p logs "${TRACE_DIR:-data/traces}" "${TRAIN_DIR:-data/training}"
echo "job ${SLURM_JOB_ID:-local} started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
.venv/bin/python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else "no GPU visible")'
