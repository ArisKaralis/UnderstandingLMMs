#!/usr/bin/env bash
# Tiny LoRA SFT with step recording (same recipe as MLLM_Testing train_lora.py).
#   sbatch -p <gpu partition> --export=ALL,MODEL=qwen3-vl-2b,DATA=vlir,SMOKE=1 scripts/hpc/train_sft.slurm
#   sbatch -p <gpu partition> --export=ALL,MODEL=qwen3-vl-2b,DATA=vlir scripts/hpc/train_sft.slurm
#   for arm in edited original; do sbatch -p <gpu partition> --export=ALL,MODEL=qwen3-vl-2b,DATA=steer_$arm,EPOCHS=5 scripts/hpc/train_sft.slurm; done
#SBATCH --job-name=lmmviz-sft
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
source scripts/hpc/_job_header.sh
MODEL="${MODEL:-qwen3-vl-2b}"; DATA="${DATA:-vlir}"
ARGS=(--model "$MODEL" --data "$DATA" --epochs "${EPOCHS:-1}" --n-train "${N_TRAIN:-500}" --run-id "sft__${MODEL}__${DATA}${SMOKE:+_smoke}")
[[ "${SMOKE:-}" == 1 ]] && ARGS+=(--steps 3 --n-train 12)
.venv/bin/python -m lmmviz.train_lab.sft "${ARGS[@]}" ${EXTRA_ARGS:-}
[[ "${SMOKE:-}" == 1 ]] && echo "SMOKE TEST PASSED"
echo "finished $(date)"
