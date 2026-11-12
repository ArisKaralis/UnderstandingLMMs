#!/usr/bin/env bash
# Download weights on the LOGIN node (compute nodes usually have no internet).
#   bash scripts/hpc/download_models.sh configs/experiments/baseline_small.yaml configs/experiments/baseline_large.yaml
#       -> exactly the models those experiment files use (recommended)
#   bash scripts/hpc/download_models.sh A      # registry tier A (all <=4B models);  B = all 8B-35B
# Models already in $HF_HOME finish in seconds.
set -euo pipefail
[[ -f scripts/hpc/cluster.env ]] && source scripts/hpc/cluster.env
unset PYTHONPATH
IDS=$(.venv/bin/python - "$@" <<'PY'
import sys, yaml
from lmmviz.specs.registry import MODELS, BY_KEY
args = sys.argv[1:] or ["AB"]
keys = []
for a in args:
    if a.endswith(".yaml"):
        keys += yaml.safe_load(open(a))["models"]
    else:
        keys += [m.key for m in MODELS if m.tier in a]
seen = []
for k in keys:
    if BY_KEY[k].hf_id not in seen:
        seen.append(BY_KEY[k].hf_id)
print(" ".join(seen))
PY
)
for id in $IDS; do
  echo "== $id"
  .venv/bin/hf download "$id" > /dev/null
done
echo "downloads done"
