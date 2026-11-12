#!/usr/bin/env bash
# Tiny RL with step recording. Run the four algorithms from the same SFT adapter and seed:
#   for a in grpo dapo gspo sapo; do
#     sbatch -p <gpu partition> --export=ALL,MODEL=qwen3-vl-2b,ALGO=$a,INIT=$TRAIN_DIR/sft__qwen3-vl-2b__vlir/adapter scripts/hpc/train_rl.slurm
#   done
#   (add SMOKE=1 first)
#SBATCH --job-name=lmmviz-rl
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
source scripts/hpc/_job_header.sh
MODEL="${MODEL:-qwen3-vl-2b}"; ALGO="${ALGO:-grpo}"
ARGS=(--model "$MODEL" --algo "$ALGO" --steps "${STEPS:-150}" --group "${GROUP:-8}" --prompts "${PROMPTS:-2}" --run-id "rl__${MODEL}__${ALGO}${SMOKE:+_smoke}")
[[ -n "${INIT:-}" ]] && ARGS+=(--init-adapter "$INIT")
[[ "${SMOKE:-}" == 1 ]] && ARGS+=(--steps 2 --group 4 --prompts 1 --max-new-tokens 128)
.venv/bin/python -m lmmviz.train_lab.rl "${ARGS[@]}" ${EXTRA_ARGS:-}
[[ "${SMOKE:-}" == 1 ]] && echo "SMOKE TEST PASSED"
echo "finished $(date)"
