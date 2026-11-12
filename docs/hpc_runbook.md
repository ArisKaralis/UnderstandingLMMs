# HPC runbook

This follows `hpc_gpu_jobs_guide.md`. The AI assistant edits code locally; **you** push, pull, `sbatch` and copy results. Cluster details live only in the gitignored `scripts/hpc/cluster.env`, and the partition is passed on the command line.

## One-time setup (login node)

```bash
cd ~ && git clone <this repo URL> UnderstandingLMMs && cd UnderstandingLMMs
# MLLM_Testing must be cloned too (VLIR loaders, crop loop, judge rules are imported from it)
cp scripts/hpc/cluster.env.example scripts/hpc/cluster.env    # set module line, HF_HOME, DATA_DIR, MLLM_TESTING_DIR
bash scripts/hpc/setup_env.sh
bash scripts/hpc/download_models.sh A     # small models first; later without the argument for all (≈330 GB)
```

The VLIR data is already on scratch from MLLM_Testing (`$DATA_DIR/VLIR`). MMVP and V* images are read from `MLLM_Testing/data/r3eval`.

## 1. Smoke test (always first)

```bash
sbatch -p <gpu partition> scripts/hpc/capture_smoke.slurm
tail -f logs/lmmviz-smoke_<jobid>.out          # ends with: SMOKE TEST PASSED
```

## 2. Captures

```bash
EXP=configs/experiments/baseline_small.yaml    # 130 runs, any GPU; baseline_large.yaml = 208 runs, 80 GB GPU (lite policy)
N=$(.venv/bin/python -m lmmviz.capture.run --experiment $EXP --list | tail -1 | cut -d' ' -f1)
sbatch -p <gpu partition> --array=0-$((N-1))%8 scripts/hpc/capture_array.slurm $EXP
```

- `--list` prints each run index and its bundle id. Use it to submit only some indices (e.g. the small models: `--array=0-51`).
- Tier-B models (8B–35B) need one 80 GB GPU. Submit those indices with a bigger-GPU partition and `--mem=160G`.
- Runs are resumable: resubmitting skips complete bundles.
- Other grids: `full_small.yaml` (every tensor for the small models), `interventions.yaml` (396 intervention runs, lite policy).

## 3. Training lab

```bash
sbatch -p <gpu partition> --export=ALL,MODEL=qwen3-vl-2b,DATA=vlir,SMOKE=1 scripts/hpc/train_sft.slurm      # smoke
sbatch -p <gpu partition> --export=ALL,MODEL=qwen3-vl-2b,DATA=vlir scripts/hpc/train_sft.slurm
for arm in edited original; do
  sbatch -p <gpu partition> --export=ALL,MODEL=qwen3-vl-2b,DATA=steer_$arm,EPOCHS=5 scripts/hpc/train_sft.slurm
done
for a in grpo dapo gspo sapo; do
  sbatch -p <gpu partition> --export=ALL,MODEL=qwen3-vl-2b,ALGO=$a,INIT=$TRAIN_DIR/sft__qwen3-vl-2b__vlir/adapter scripts/hpc/train_rl.slurm
done
sbatch -p <gpu partition> --export=ALL,SMOKE=1 scripts/hpc/train_distill.slurm
```

RL uses `$DATA_DIR/VLIR/rl/data_rgrpo.json` rows that have a gold box. If the file's field names differ from what `lmmviz/train_lab/rl.py:load_prompts` expects, it falls back to VLIR SFT rows. Check the first log line for `n_prompts`.

## 4. Bring results back (you, on the laptop)

Copy the bundle folders from `$TRACE_DIR` into `data/traces/` and the run folders from `$TRAIN_DIR` into `data/training/` (VS Code "Download", or your own rsync/scp). Then:

```bash
.venv/bin/python scripts/local/validate_bundle.py data/traces/*
.venv/bin/uvicorn backend.app.main:app --port 8000
```

## When a job fails

Paste only the relevant error lines, with hostnames and paths removed.

## Sizes to expect

| | per bundle (standard) |
|---|---|
| ≤2B, 400–600 tokens | 60–140 MB |
| 8B–9B | ~250 MB |
| 27B–35B | ~0.5–1 GB (use `lite` for big grids) |
