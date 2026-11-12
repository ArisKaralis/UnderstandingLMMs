#!/usr/bin/env bash
# Download weights on the LOGIN node (compute nodes usually have no internet).
#   bash scripts/hpc/download_models.sh        # tiers A + B: every in-scope model (<=100B), ~330 GB
#   bash scripts/hpc/download_models.sh A      # only the small tier-A models
# Models already in $HF_HOME (e.g. Qwen3-VL-8B, Qwen3.5-9B, Qwen3.8-27B from MLLM_Testing) finish quickly.
set -euo pipefail
[[ -f scripts/hpc/cluster.env ]] && source scripts/hpc/cluster.env
TIER="${1:-AB}"
IDS=$(.venv/bin/python -c "from lmmviz.specs.registry import MODELS; print(' '.join(m.hf_id for m in MODELS if m.tier in '$TIER'))")
for id in $IDS; do
  echo "== $id"
  .venv/bin/hf download "$id" > /dev/null
done
echo "downloads done"
