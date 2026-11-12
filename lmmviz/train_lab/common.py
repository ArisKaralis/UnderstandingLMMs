"""Shared pieces for the tiny-scale training runs (SFT, RL, distillation).

Runs write `data/training/<run_id>/`:
    run.json        what was run (kind, model, algo, config, data, git)
    metrics.jsonl   one line per optimiser step
    steps/<k>.json  full per-token record of selected steps (what the dashboard replays)
    images/         prompt images referenced by the step files
    eval.json       before/after answers on held-out items (when evaluated)
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

from lmmviz.data import vlir
from lmmviz.specs.registry import BY_KEY

OUT_ROOT = Path(os.environ.get("TRAIN_DIR", "data/training"))


def pick_device() -> str:
    from lmmviz.capture.run import pick_device as p
    return p()


def load(model_key: str, device: str, max_pixels: int):
    from lmmviz.capture.run import load_model
    return load_model(BY_KEY[model_key].hf_id, device, max_pixels)


def train_lora_module():
    """MLLM_Testing/scripts/sft/train_lora.py (VLIRDataset, collate, LORA_TARGETS) — reused, not copied."""
    vlir.mllm()  # puts MLLM_Testing/scripts on sys.path
    import importlib
    return importlib.import_module("sft.train_lora")


def lora_config(r: int = 64, alpha: int = 128, dropout: float = 0.05):
    from peft import LoraConfig
    tl = train_lora_module()
    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=tl.LORA_TARGETS, task_type="CAUSAL_LM")


def vlir_sft_rows(split: str = "train", limit: int | None = None, require_local_images: bool = True, seed: int = 0) -> list[dict]:
    """VLIR SFT rows in MLLM_Testing's training format (make_data.convert), split by make_data.is_dev."""
    md = vlir.mllm()["make_data"]
    pre = vlir.mllm_dir() / "data" / "sft" / f"vlir_qwen3_{split}.jsonl"
    if pre.exists():
        rows = [json.loads(l) for l in pre.open()]
    else:
        raw = json.loads((vlir.vlir_root() / "sft" / "vlir_sft_12k.json").read_text())
        rows = []
        for r in raw:
            c = md.convert(r)
            if c is None:
                continue
            if md.is_dev(c["image"], 0.02) != (split == "dev"):
                continue
            rows.append(c)
    root = data_root()
    if require_local_images:
        rows = [r for r in rows if (root / r["image"]).exists()]
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def steer_rows(arm: str) -> list[dict]:
    p = vlir.mllm_dir() / "data" / "vlir_steer" / f"train_{arm}.jsonl"
    return [json.loads(l) for l in p.open()]


def data_root() -> Path:
    """Folder that holds VLIR/ (row['image'] = 'VLIR/sft/imgs/...')."""
    return vlir.vlir_root().parent


def write_rows(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def git_rev() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


class RunWriter:
    def __init__(self, run_id: str, meta: dict):
        self.dir = OUT_ROOT / run_id
        (self.dir / "steps").mkdir(parents=True, exist_ok=True)
        (self.dir / "images").mkdir(exist_ok=True)
        self.meta = {**meta, "id": run_id, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": git_rev(),
                     "mllm_testing": vlir.mllm_commit(), "torch": torch.__version__}
        (self.dir / "run.json").write_text(json.dumps(self.meta, indent=1))
        self.metrics = (self.dir / "metrics.jsonl").open("w")

    def metric(self, row: dict) -> None:
        self.metrics.write(json.dumps(row) + "\n")
        self.metrics.flush()
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

    def step(self, k: int, data: dict) -> None:
        (self.dir / "steps" / f"{k:05d}.json").write_text(json.dumps(data))

    def image(self, name: str, img) -> str:
        p = self.dir / "images" / f"{re.sub(r'[^A-Za-z0-9._-]', '_', name)}.png"
        if not p.exists():
            img.convert("RGB").save(p)
        return str(p.relative_to(self.dir))

    def finish(self, extra: dict | None = None) -> None:
        self.metrics.close()
        if extra:
            self.meta.update(extra)
            (self.dir / "run.json").write_text(json.dumps(self.meta, indent=1))


def record_steps(total: int) -> set[int]:
    """Which optimiser steps get a full per-token record."""
    return {0, 1, 2, 5, 10, 20, 50, 100, 200, total - 1} & set(range(total))


def module_group(name: str) -> str:
    m = re.search(r"layers\.(\d+)\.(self_attn|linear_attn|mlp)", name)
    if m:
        return f"L{int(m.group(1)):02d}.{ {'self_attn': 'attn', 'linear_attn': 'gdn', 'mlp': 'mlp'}[m.group(2)] }"
    if "visual" in name:
        return "vision"
    if "embed" in name:
        return "embed"
    if "lm_head" in name:
        return "lm_head"
    return "other"


def grad_norms_by_group(model) -> dict[str, float]:
    acc: dict[str, float] = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            g = module_group(n)
            acc[g] = acc.get(g, 0.0) + float(p.grad.detach().float().pow(2).sum())
    return {k: v ** 0.5 for k, v in sorted(acc.items())}


def lora_delta_norms(model) -> dict[str, list[float]]:
    """‖ΔW‖_F = scale·‖B A‖_F per target projection per layer."""
    out: dict[str, dict[int, float]] = {}
    for n, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B") and "default" in getattr(mod, "lora_A", {}):
            m = re.search(r"layers\.(\d+)\..*\.(\w+_proj)$", n)
            if not m:
                continue
            A, B = mod.lora_A["default"].weight, mod.lora_B["default"].weight
            d = (B.float() @ A.float()) * mod.scaling["default"]
            out.setdefault(m.group(2), {})[int(m.group(1))] = float(d.norm())
    return {k: [v.get(i, 0.0) for i in range(max(v) + 1)] for k, v in out.items()}
