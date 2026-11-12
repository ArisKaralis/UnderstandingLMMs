# Inside Qwen-VL

An interactive dashboard for understanding how the Qwen-VL family of multimodal models works:
**Qwen3-VL, Qwen3.5 and Qwen3.8** (open models up to 100B parameters).

- **Specs**: every model's configuration, derived parameter counts (equal to the published safetensors totals), KV-cache/state sizes, and a lineage timeline.
- **Architecture**: the model as a zoomable graph built from its real `config.json`. Click a stage, then a layer, then an operation. Tensor shapes, pseudo-code, parameters and FLOPs sit in drop-downs that are closed by default. The **config playground** lets you edit the architecture and see shapes, parameters and cost update.
- **Trace**: one real input (your VLIR / V* / MMVP samples) followed through a captured forward pass. Stages: preprocessing → tokens and 3-D MRoPE positions → ViT attention → merger / DeepStack → per-layer residual stream, attention, gated-attention gates, Gated DeltaNet memory, MoE routing → logit lens → generated tokens.
- **Experiments**: baseline vs intervention (head ablation, DeepStack off, Gated DeltaNet without forgetting, attention gate off, flat positions, resolution, blank image, fewer experts…).
- **Training lab**: pre- and post-training pipelines from the papers, a GRPO / DAPO / GSPO / SAPO playground, and step-by-step replays of real tiny SFT and RL runs on VLIR.

```
HPC (Slurm, you run)                     laptop
 lmmviz.capture.run  ──► trace bundles ──►  backend (FastAPI) ──► frontend (React + D3)
 lmmviz.train_lab.*  ──► training runs ──►  data/traces, data/training
```

## Quick start (laptop)

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-train.txt -r requirements-app.txt && .venv/bin/pip install -e .
python -m lmmviz.specs.download_configs     # config.json files only (already committed in data/specs/configs)
python scripts/local/make_demo_bundle.py     # tiny random-weight bundles, no weights needed
cd frontend && npm install && npm run build && cd ..
.venv/bin/uvicorn backend.app.main:app --port 8000   # → http://localhost:8000
# frontend dev server with hot reload: cd frontend && npm run dev  (→ :5173, proxies /api)
```

Real captures of small models also run on Apple Silicon (MPS):

```bash
python -m lmmviz.capture.run --model qwen3.5-0.8b --sample vlir-152 --mode direct
python -m lmmviz.capture.run --experiment configs/experiments/local_demo.yaml      # baseline + 7 interventions
python -m lmmviz.train_lab.sft --model qwen3.5-0.8b --n-train 12 --steps 6 --grad-accum 2 --demo
python -m lmmviz.train_lab.rl  --model qwen3.5-0.8b --algo grpo --source sft --group 4 --prompts 1 --steps 4 --demo
```

`../MLLM_Testing` must sit next to this repo (or set `MLLM_TESTING_DIR`). The VLIR loaders, the VLM-R³ crop loop, the judge rules and the LoRA dataset are imported from it, not copied.

## HPC

See [docs/hpc_runbook.md](docs/hpc_runbook.md). In short: `git pull` → `bash scripts/hpc/setup_env.sh` → `bash scripts/hpc/download_models.sh` → `sbatch -p <partition> scripts/hpc/capture_smoke.slurm` → array jobs over `configs/experiments/*.yaml` → copy bundles to `data/traces/` → `python scripts/local/validate_bundle.py data/traces/*`.

## Layout

| Path | What |
|---|---|
| `lmmviz/specs/` | model registry (≤100B), config download, parameter / memory / FLOP derivation |
| `lmmviz/graph/build_graph.py` | config → architecture graph with symbolic + concrete shapes (also serves the playground) |
| `lmmviz/capture/` | hooks / recorders, capture policies, bundle writer, runner (direct + R³ crop loop) |
| `lmmviz/interventions/registry.py` | reversible model / input edits used by experiments |
| `lmmviz/train_lab/` | `losses.py` (GRPO, DAPO, GSPO, SAPO), `sft.py`, `rl.py`, `distill.py`, `rewards.py` |
| `lmmviz/data/` | VLIR / benchmark sample loading, shared with MLLM_Testing |
| `backend/app/` | FastAPI: specs, graph, traces, previews, tensors, diff, training |
| `frontend/src/` | pages (Specs, Architecture, Trace, Experiments, Training, Learn), charts, content |
| `configs/experiments/` | capture grids for the HPC |
| `docs/` | bundle format, HPC runbook, literature notes |

## Tests

```bash
.venv/bin/pytest -q                                  # specs vs HF totals, graph, losses, capture, interventions, API
cd frontend && npx tsc -p . && node e2e/smoke.mjs     # type-check + headless click-through (server must be running)
```
