"""The set of models the app covers.

Scope rule: only models with <= 100B total parameters (MoE included). Larger releases
(Qwen3-VL-235B-A22B, Qwen3.5-122B-A10B / 397B-A17B, Qwen3.8-Flash-Next, Qwen3.8-2.4T-A95B)
are listed in LINEAGE_NOTES only, never captured or rendered as architectures.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class ModelEntry:
    key: str            # short id used in URLs / bundle names
    hf_id: str          # Hugging Face repo
    family: str         # "qwen3_vl" | "qwen3_5" | "qwen3_8"
    label: str
    released: str       # YYYY-MM-DD (first release of this size)
    license: str
    variant: str = "instruct"   # instruct | thinking | hybrid
    tier: str = "A"             # HPC tier (A: any 1 GPU, B: 1x80GB)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


MODELS: list[ModelEntry] = [
    # ---- Qwen3-VL (arXiv 2511.21631) ----
    ModelEntry("qwen3-vl-2b", "Qwen/Qwen3-VL-2B-Instruct", "qwen3_vl", "Qwen3-VL-2B-Instruct", "2025-10-21", "apache-2.0", tier="A"),
    ModelEntry("qwen3-vl-4b", "Qwen/Qwen3-VL-4B-Instruct", "qwen3_vl", "Qwen3-VL-4B-Instruct", "2025-10-15", "apache-2.0", tier="A"),
    ModelEntry("qwen3-vl-8b", "Qwen/Qwen3-VL-8B-Instruct", "qwen3_vl", "Qwen3-VL-8B-Instruct", "2025-10-15", "apache-2.0", tier="B"),
    ModelEntry("qwen3-vl-8b-thinking", "Qwen/Qwen3-VL-8B-Thinking", "qwen3_vl", "Qwen3-VL-8B-Thinking", "2025-10-15", "apache-2.0", variant="thinking", tier="B"),
    ModelEntry("qwen3-vl-32b", "Qwen/Qwen3-VL-32B-Instruct", "qwen3_vl", "Qwen3-VL-32B-Instruct", "2025-10-21", "apache-2.0", tier="B"),
    ModelEntry("qwen3-vl-30b-a3b", "Qwen/Qwen3-VL-30B-A3B-Instruct", "qwen3_vl", "Qwen3-VL-30B-A3B-Instruct", "2025-10-04", "apache-2.0", tier="B"),
    # ---- Qwen3.5 (native multimodal, hybrid Gated DeltaNet) ----
    ModelEntry("qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B", "qwen3_5", "Qwen3.5-0.8B", "2026-03-02", "apache-2.0", variant="hybrid", tier="A"),
    ModelEntry("qwen3.5-2b", "Qwen/Qwen3.5-2B", "qwen3_5", "Qwen3.5-2B", "2026-03-02", "apache-2.0", variant="hybrid", tier="A"),
    ModelEntry("qwen3.5-4b", "Qwen/Qwen3.5-4B", "qwen3_5", "Qwen3.5-4B", "2026-03-02", "apache-2.0", variant="hybrid", tier="A"),
    ModelEntry("qwen3.5-9b", "Qwen/Qwen3.5-9B", "qwen3_5", "Qwen3.5-9B", "2026-03-02", "apache-2.0", variant="hybrid", tier="B"),
    ModelEntry("qwen3.5-27b", "Qwen/Qwen3.5-27B", "qwen3_5", "Qwen3.5-27B", "2026-02-24", "apache-2.0", variant="hybrid", tier="B"),
    ModelEntry("qwen3.5-35b-a3b", "Qwen/Qwen3.5-35B-A3B", "qwen3_5", "Qwen3.5-35B-A3B", "2026-02-24", "apache-2.0", variant="hybrid", tier="B"),
    # ---- Qwen3.8 (only the <=100B open VLM) ----
    ModelEntry("qwen3.8-27b", "Qwen/Qwen3.8-27B", "qwen3_8", "Qwen3.8-27B", "2026-08-14", "apache-2.0", variant="hybrid", tier="B",
               notes=["Same model class as Qwen3.5 (Qwen3_5ForConditionalGeneration); builds on Qwen3.6-27B.",
                      "reasoning_effort (xhigh/medium/low) and preserve_thinking inference controls."]),
]

BY_KEY = {m.key: m for m in MODELS}

FAMILIES = {
    "qwen3_vl": {"label": "Qwen3-VL", "date": "2025-11", "paper": "https://arxiv.org/abs/2511.21631"},
    "qwen3_5": {"label": "Qwen3.5", "date": "2026-02", "paper": "https://huggingface.co/collections/Qwen/qwen35"},
    "qwen3_8": {"label": "Qwen3.8", "date": "2026-08", "paper": "https://github.com/QwenLM/Qwen3.8"},
}

# Shown on the Specs page only as text; out of scope (>100B) for capture and diagrams.
LINEAGE_NOTES = [
    {"date": "2025-02", "name": "Qwen2.5-VL", "change": "Window-attention ViT (patch 14), MRoPE aligned to absolute time; SFT + DPO post-training."},
    {"date": "2025-11", "name": "Qwen3-VL", "change": "SigLIP-2 ViT (patch 16), DeepStack (multi-level ViT features added to LLM layers 0-2), interleaved MRoPE, text timestamps for video; SAPO RL."},
    {"date": "2026-02", "name": "Qwen3.5", "change": "Early-fusion native multimodal; hybrid 3:1 Gated DeltaNet / gated attention; DeepStack removed; vocab 248K; MTP; one hybrid thinking model."},
    {"date": "2026-04", "name": "Qwen3.6", "change": "27B dense & 35B-A3B on the same architecture; thinking preservation introduced."},
    {"date": "2026-08", "name": "Qwen3.8", "change": "27B dense VLM on the Qwen3.5 architecture; reasoning_effort levels; preserve_thinking on by default."},
    {"date": "2026-08", "name": "Out of scope (>100B)", "change": "Qwen3-VL-235B-A22B, Qwen3.5-122B-A10B / 397B-A17B, Qwen3.8-Flash-Next (125B-A6B, new QSA / Gated Residual / n-gram design), Qwen3.8-2.4T-A95B."},
]
