"""Policy-gradient objectives used by the Training Lab, in one place.

Inputs for one prompt group (G sampled responses, padded to length L):
    logp      [G, L]  log π_θ(y_t | x, y_<t)      (requires grad during training)
    logp_old  [G, L]  log π_θ_old (behaviour policy that sampled the rollouts)
    logp_ref  [G, L]  log π_ref  (frozen reference, for the KL term)   optional
    mask      [G, L]  1 for response tokens that get loss
    rewards   [G]

Every intermediate is returned so the dashboard can show, token by token, what each algorithm does.
References: GRPO (arXiv 2402.03300), DAPO (2503.14476), GSPO (2507.18071), SAPO (2511.20347).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch


@dataclass
class AlgoConfig:
    algo: str = "grpo"            # grpo | dapo | gspo | sapo
    eps_low: float = 0.2          # GRPO/DAPO clip; GSPO uses 3e-4
    eps_high: float = 0.2         # DAPO "clip-higher": 0.28; GSPO: 4e-4
    beta_kl: float = 0.04         # GRPO KL weight (DAPO/GSPO/SAPO default 0)
    tau_pos: float = 1.0          # SAPO temperature for positive advantages
    tau_neg: float = 1.05         # SAPO temperature for negative advantages
    std_normalize: bool = True
    token_level: bool = False     # DAPO: normalise by total tokens in the group, not per sequence

    @classmethod
    def preset(cls, algo: str) -> "AlgoConfig":
        return {
            "grpo": cls("grpo", 0.2, 0.2, 0.04),
            "dapo": cls("dapo", 0.2, 0.28, 0.0, token_level=True),
            "gspo": cls("gspo", 3e-4, 4e-4, 0.0),
            "sapo": cls("sapo", 0.2, 0.2, 0.0, 1.0, 1.05),
        }[algo]

    def to_dict(self) -> dict:
        return asdict(self)


def group_advantages(rewards: torch.Tensor, std_normalize: bool = True) -> torch.Tensor:
    a = rewards - rewards.mean()
    if std_normalize:
        a = a / (rewards.std(unbiased=False) + 1e-6) if rewards.numel() > 1 else a
    return a


def k3_kl(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """Unbiased, non-negative per-token KL estimator used by GRPO: r - log r - 1, r = π_ref/π."""
    d = logp_ref - logp
    return d.exp() - d - 1


def objective(logp, logp_old, mask, rewards, cfg: AlgoConfig, logp_ref=None) -> dict:
    mask = mask.float()
    adv = group_advantages(rewards.float(), cfg.std_normalize)          # [G]
    A = adv[:, None].expand_as(logp)                                     # every token of y_i shares Â_i
    log_ratio = (logp - logp_old) * mask
    ratio = log_ratio.exp()                                              # r_{i,t}
    lens = mask.sum(1).clamp(min=1)
    out = {"advantages": adv, "ratio": ratio, "mask": mask}

    if cfg.algo in ("grpo", "dapo"):
        clipped = ratio.clamp(1 - cfg.eps_low, 1 + cfg.eps_high)
        surr = torch.minimum(ratio * A, clipped * A)
        # token is "clipped" when the clipped branch is active (its gradient is zero)
        out["clipped"] = ((ratio * A) > (clipped * A)).float() * mask
        per_tok = surr
    elif cfg.algo == "gspo":
        s = (log_ratio.sum(1) / lens).exp()                              # s_i = (π/π_old)^(1/|y|)
        # GSPO-token form: value s_i per token, gradient of log π_t (sg = stop-gradient)
        s_tok = s.detach()[:, None] * (logp - logp.detach()).exp()
        s_clip = s_tok.clamp(1 - cfg.eps_low, 1 + cfg.eps_high)
        per_tok = torch.minimum(s_tok * A, s_clip * A)
        seq_clipped = ((s * adv) > (s.clamp(1 - cfg.eps_low, 1 + cfg.eps_high) * adv)).float()
        out["seq_ratio"] = s
        out["clipped"] = seq_clipped[:, None] * mask
    elif cfg.algo == "sapo":
        tau = torch.where(A > 0, torch.full_like(A, cfg.tau_pos), torch.full_like(A, cfg.tau_neg))
        f = (4.0 / tau) * torch.sigmoid(tau * (ratio - 1))                 # soft gate replaces the hard clip
        per_tok = f * A
        sig = torch.sigmoid(tau * (ratio - 1))
        out["gate_weight"] = (4 * sig * (1 - sig)).detach() * mask       # = sech²(τ(r-1)/2), 1 at r=1
        out["clipped"] = (out["gate_weight"] < 0.5).float() * mask        # "softly clipped": weight below 1/2
    else:
        raise ValueError(cfg.algo)

    if logp_ref is not None and cfg.beta_kl > 0:
        kl = k3_kl(logp, logp_ref) * mask
        per_tok = per_tok - cfg.beta_kl * kl
        out["kl"] = kl
    per_tok = per_tok * mask
    if cfg.token_level:
        J = per_tok.sum() / mask.sum().clamp(min=1)
    else:
        J = (per_tok.sum(1) / lens).mean()
    out["per_token_objective"] = per_tok
    out["objective"] = J
    out["loss"] = -J
    out["clip_fraction"] = out["clipped"].sum() / mask.sum().clamp(min=1)
    return out


def dapo_keep_group(rewards: torch.Tensor) -> bool:
    """Dynamic sampling: drop groups whose rewards are all equal (zero advantage → no signal)."""
    return bool((rewards.max() - rewards.min()) > 1e-8)


def soft_overlong_penalty(length: int, l_max: int = 16384, l_cache: int = 4096) -> float:
    if length <= l_max - l_cache:
        return 0.0
    if length <= l_max:
        return ((l_max - l_cache) - length) / l_cache
    return -1.0


def recompute_all(logp, logp_old, mask, rewards, logp_ref=None) -> dict[str, dict]:
    """Same stored step under every algorithm (for the dashboard toggle)."""
    res = {}
    for a in ("grpo", "dapo", "gspo", "sapo"):
        o = objective(logp, logp_old, mask, rewards, AlgoConfig.preset(a), logp_ref)
        res[a] = {k: (v.detach().cpu().tolist() if torch.is_tensor(v) else v) for k, v in o.items()}
        res[a]["config"] = AlgoConfig.preset(a).to_dict()
    return res
