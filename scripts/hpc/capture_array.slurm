#!/usr/bin/env bash
# One Slurm array task per run of an experiment YAML. Resumable: complete bundles are skipped.
#   N=$(.venv/bin/python -m lmmviz.capture.run --experiment configs/experiments/baseline.yaml --list | tail -1 | cut -d' ' -f1)
#   sbatch -p <gpu partition> --array=0-$((N-1))%8 scripts/hpc/capture_array.slurm configs/experiments/baseline.yaml
# Memory: tier-A models fit any GPU; tier B (8B-35B) wants one 80 GB GPU. For a tier-B-only YAML use
#   sbatch -p <gpu partition> --gres=gpu:1 --mem=160G ...   (override on the command line)
#SBATCH --job-name=lmmviz-capture
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
source scripts/hpc/_job_header.sh
EXP="${1:?usage: sbatch ... capture_array.slurm <experiment.yaml>}"
.venv/bin/python -m lmmviz.capture.run --experiment "$EXP" --index "${SLURM_ARRAY_TASK_ID:?run as an array job}"
echo "finished $(date)"
