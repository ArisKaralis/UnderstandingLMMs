"""Capture policies: how much of each run is stored.

A 36-layer, 4096-wide model on a 1.5K-token sequence produces ~0.4 GB of residual stream alone,
and a full attention map is H*T*T per layer, so most tensors are summarised (stats + previews)
and only selected ones are stored in full.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Policy:
    name: str
    full_tensors: str            # "none" | "selected" | "all"
    vision_attn_layers: str      # "taps" (deepstack/first/mid/last) | "all"
    text_attn_layers: str        # "all" | "every4"
    gdn_state: bool = True       # recompute the delta-rule recurrence for state norms
    logit_lens: bool = True
    pca: bool = True
    hist_bins: int = 32
    max_attn_tokens: int = 4096  # skip head-mean T×T matrices above this length (keep last-row only)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


POLICIES = {
    "lite": Policy("lite", full_tensors="none", vision_attn_layers="taps", text_attn_layers="every4"),
    "standard": Policy("standard", full_tensors="selected", vision_attn_layers="taps", text_attn_layers="all"),
    "full": Policy("full", full_tensors="all", vision_attn_layers="all", text_attn_layers="all"),
}


def get(name: str) -> Policy:
    return POLICIES[name]


def selected_layers(n: int, k: int = 2) -> set[int]:
    """first k, middle k, last k layers."""
    mid = n // 2
    return set(range(min(k, n))) | set(range(max(0, mid - k // 2), min(n, mid + (k + 1) // 2))) | set(range(max(0, n - k), n))
