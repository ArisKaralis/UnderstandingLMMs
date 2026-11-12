"""Summary statistics for any tensor (one row of stats.parquet)."""
from __future__ import annotations

import numpy as np
import torch


def tensor_stats(name: str, node: str, t: torch.Tensor, bins: int = 32) -> dict:
    x = t.detach().float()
    flat = x.reshape(-1)
    n = flat.numel()
    finite = torch.isfinite(flat)
    f = flat[finite] if not bool(finite.all()) else flat
    if f.numel() == 0:
        f = torch.zeros(1, device=flat.device)
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        hi = lo + 1e-6
    hist = torch.histc(f, bins=bins, min=lo, max=hi).cpu().numpy().astype(np.int64)
    row = {
        "name": name, "node": node, "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
        "numel": int(n), "mean": float(f.mean()), "std": float(f.std()) if f.numel() > 1 else 0.0,
        "min": lo, "max": float(f.max()), "absmax": float(f.abs().max()), "l2": float(f.norm()),
        "frac_zero": float((f == 0).float().mean()), "n_nonfinite": int(n - int(finite.sum())),
        "hist": hist.tolist(), "hist_lo": lo, "hist_hi": hi,
    }
    if x.dim() >= 2:  # per-row (token) L2 norms, used for layer×token heatmaps
        rows = x.reshape(-1, x.shape[-1]) if x.dim() == 2 else x.reshape(x.shape[0] if x.dim() == 2 else -1, x.shape[-1])
        if rows.shape[0] <= 65536:
            row["row_norm_mean"] = float(rows.norm(dim=-1).mean())
    return row
