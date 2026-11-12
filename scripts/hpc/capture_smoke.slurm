#!/usr/bin/env bash
# 1 small model x 1 sample, lite + standard policy. Run this first.
#   sbatch -p <gpu partition> scripts/hpc/capture_smoke.slurm
#SBATCH --job-name=lmmviz-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
source scripts/hpc/_job_header.sh
OUT="${TRACE_DIR:-data/traces}_smoke"
.venv/bin/python -m lmmviz.capture.run --model qwen3.5-0.8b --sample vlir-152 --mode direct --policy lite --max-new-tokens 32 --out "$OUT" --force
.venv/bin/python -m lmmviz.capture.run --model qwen3-vl-2b --sample vstar-1 --mode r3 --policy standard --max-new-tokens 256 --out "$OUT" --force
.venv/bin/python scripts/local/validate_bundle.py "$OUT"/*
echo "SMOKE TEST PASSED"
