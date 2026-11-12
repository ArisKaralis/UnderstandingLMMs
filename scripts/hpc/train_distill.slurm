#!/usr/bin/env bash
# On-policy distillation Qwen3-VL-8B -> Qwen3-VL-2B (both on one 80 GB GPU).
#   sbatch -p <gpu partition> --export=ALL,SMOKE=1 scripts/hpc/train_distill.slurm
#SBATCH --job-name=lmmviz-distill
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
source scripts/hpc/_job_header.sh
ARGS=(--student "${STUDENT:-qwen3-vl-2b}" --teacher "${TEACHER:-qwen3-vl-8b}" --steps "${STEPS:-100}")
[[ "${SMOKE:-}" == 1 ]] && ARGS+=(--steps 2 --max-new-tokens 64 --run-id distill_smoke)
.venv/bin/python -m lmmviz.train_lab.distill "${ARGS[@]}"
[[ "${SMOKE:-}" == 1 ]] && echo "SMOKE TEST PASSED"
echo "finished $(date)"
