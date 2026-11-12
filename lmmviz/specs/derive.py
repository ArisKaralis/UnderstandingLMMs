"""Normalise HF configs into one schema and derive parameter counts, memory and FLOPs.

The parameter formulas mirror the module definitions in transformers 5.17
(models/qwen3_vl, qwen3_vl_moe, qwen3_5, qwen3_5_moe). Every number the dashboard shows
about model size comes from here, so it is unit-tested against Hugging Face's safetensors totals.

    python -m lmmviz.specs.derive            # writes data/specs/models.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .registry import MODELS, FAMILIES, LINEAGE_NOTES, BY_KEY

CONFIG_DIR = Path("data/specs/configs")


# --------------------------------------------------------------------------- normalise
def _rope(tc: dict) -> dict:
    rp = tc.get("rope_parameters") or tc.get("rope_scaling") or {}
    theta = rp.get("rope_theta", tc.get("rope_theta"))
    prf = rp.get("partial_rotary_factor", tc.get("partial_rotary_factor", 1.0))
    return {
        "theta": theta,
        "partial_rotary_factor": prf,
        "mrope_section": rp.get("mrope_section"),
        "mrope_interleaved": bool(rp.get("mrope_interleaved", False)),
        "type": rp.get("rope_type", "default"),
    }


def normalise(cfg: dict, family: str) -> dict:
    tc = cfg.get("text_config", cfg)
    vc = cfg.get("vision_config")
    L = tc["num_hidden_layers"]
    d = tc["hidden_size"]
    head_dim = tc.get("head_dim") or d // tc["num_attention_heads"]
    layer_types = tc.get("layer_types") or ["full_attention"] * L
    rope = _rope(tc)
    rotary_dim = int(head_dim * rope["partial_rotary_factor"])
    moe = None
    if tc.get("num_experts"):
        moe = {
            "num_experts": tc["num_experts"],
            "top_k": tc["num_experts_per_tok"],
            "expert_ffn": tc["moe_intermediate_size"],
            "shared_ffn": tc.get("shared_expert_intermediate_size") or 0,
            "norm_topk_prob": tc.get("norm_topk_prob", True),
        }
    gdn = None
    if "linear_attention" in layer_types:
        gdn = {
            "num_k_heads": tc["linear_num_key_heads"],
            "num_v_heads": tc["linear_num_value_heads"],
            "k_head_dim": tc["linear_key_head_dim"],
            "v_head_dim": tc["linear_value_head_dim"],
            "conv_kernel": tc["linear_conv_kernel_dim"],
        }
    tie = cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings", False))
    text = {
        "num_layers": L,
        "hidden_size": d,
        "intermediate_size": tc.get("intermediate_size") if not moe else None,
        "num_q_heads": tc["num_attention_heads"],
        "num_kv_heads": tc["num_key_value_heads"],
        "head_dim": head_dim,
        "rotary_dim": rotary_dim,
        "vocab_size": tc["vocab_size"],
        "max_position_embeddings": tc["max_position_embeddings"],
        "tie_word_embeddings": bool(tie),
        "rms_norm_eps": tc.get("rms_norm_eps"),
        "hidden_act": tc.get("hidden_act", "silu"),
        "attn_output_gate": bool(tc.get("attn_output_gate", False)),
        "qk_norm": True,  # Qwen3 and later always normalise q,k per head
        "layer_types": layer_types,
        "rope": rope,
        "moe": moe,
        "gdn": gdn,
        "mtp_layers": tc.get("mtp_num_hidden_layers", 0) or 0,
    }
    vision = None
    if vc:
        vision = {
            "depth": vc["depth"],
            "hidden_size": vc["hidden_size"],
            "intermediate_size": vc["intermediate_size"],
            "num_heads": vc["num_heads"],
            "head_dim": vc["hidden_size"] // vc["num_heads"],
            "patch_size": vc["patch_size"],
            "spatial_merge_size": vc["spatial_merge_size"],
            "temporal_patch_size": vc["temporal_patch_size"],
            "in_channels": vc.get("in_channels", 3),
            "out_hidden_size": vc["out_hidden_size"],
            "num_position_embeddings": vc["num_position_embeddings"],
            "deepstack_visual_indexes": vc.get("deepstack_visual_indexes") or [],
            "hidden_act": vc.get("hidden_act"),
        }
    tokens = {k: cfg.get(k) for k in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")}
    return {"family": family, "architecture": (cfg.get("architectures") or ["?"])[0], "text": text, "vision": vision, "tokens": tokens}


# --------------------------------------------------------------------------- params
def full_attn_params(t: dict) -> int:
    d, hq, hkv, hd = t["hidden_size"], t["num_q_heads"], t["num_kv_heads"], t["head_dim"]
    q_out = hq * hd * (2 if t["attn_output_gate"] else 1)  # Qwen3.5 packs the sigmoid gate into q_proj
    return d * q_out + 2 * d * hkv * hd + hq * hd * d + 2 * hd  # + q_norm, k_norm


def gdn_params(t: dict) -> int:
    g, d = t["gdn"], t["hidden_size"]
    key_dim = g["num_k_heads"] * g["k_head_dim"]
    val_dim = g["num_v_heads"] * g["v_head_dim"]
    conv_dim = 2 * key_dim + val_dim
    return (d * conv_dim            # in_proj_qkv
            + d * val_dim           # in_proj_z (output gate)
            + 2 * d * g["num_v_heads"]  # in_proj_b (beta), in_proj_a (alpha)
            + conv_dim * g["conv_kernel"]  # depthwise short conv
            + 2 * g["num_v_heads"]  # dt_bias, A_log
            + g["v_head_dim"]       # gated RMSNorm weight
            + val_dim * d)          # out_proj


def ffn_params(t: dict) -> dict:
    d = t["hidden_size"]
    if not t["moe"]:
        p = 3 * d * t["intermediate_size"]
        return {"total": p, "active": p}
    m = t["moe"]
    expert = 3 * d * m["expert_ffn"]
    router = m["num_experts"] * d
    shared = (3 * d * m["shared_ffn"] + d) if m["shared_ffn"] else 0
    return {"total": router + m["num_experts"] * expert + shared,
            "active": router + m["top_k"] * expert + shared}


def vision_params(v: dict) -> dict:
    dv, f = v["hidden_size"], v["intermediate_size"]
    patch = v["in_channels"] * v["temporal_patch_size"] * v["patch_size"] ** 2 * dv + dv
    pos = v["num_position_embeddings"] * dv
    block = (dv * 3 * dv + 3 * dv) + (dv * dv + dv) + (dv * f + f) + (f * dv + dv) + 4 * dv
    m4 = dv * v["spatial_merge_size"] ** 2
    merger_fc = (m4 * m4 + m4) + (m4 * v["out_hidden_size"] + v["out_hidden_size"])
    merger = 2 * dv + merger_fc
    deepstack = len(v["deepstack_visual_indexes"]) * (2 * m4 + merger_fc)
    total = patch + pos + v["depth"] * block + merger + deepstack
    return {"patch_embed": patch, "pos_embed": pos, "per_block": block, "blocks": v["depth"] * block,
            "merger": merger, "deepstack_mergers": deepstack, "total": total}


def text_params(t: dict) -> dict:
    d, V = t["hidden_size"], t["vocab_size"]
    attn, lin = full_attn_params(t), (gdn_params(t) if t["gdn"] else 0)
    ffn = ffn_params(t)
    n_full = sum(1 for x in t["layer_types"] if x == "full_attention")
    n_lin = t["num_layers"] - n_full
    norms = 2 * d
    layers_total = n_full * attn + n_lin * lin + t["num_layers"] * (ffn["total"] + norms)
    layers_active = n_full * attn + n_lin * lin + t["num_layers"] * (ffn["active"] + norms)
    emb = V * d
    head = 0 if t["tie_word_embeddings"] else V * d
    # MTP (Qwen3.5/3.8): fc 2d->d, 3 norms, one full-attention decoder layer; shares embedding + head.
    mtp = t["mtp_layers"] * (2 * d * d + 3 * d + attn + ffn["total"] + norms) if t["mtp_layers"] else 0
    return {"embedding": emb, "lm_head": head, "full_attn_per_layer": attn, "gdn_per_layer": lin,
            "ffn_per_layer": ffn["total"], "ffn_active_per_layer": ffn["active"],
            "n_full_attn": n_full, "n_gdn": n_lin, "layers": layers_total,
            "total": emb + head + layers_total + d, "active": emb + head + layers_active + d, "mtp": mtp}


# --------------------------------------------------------------------------- memory / flops
def per_token_costs(s: dict, dtype_bytes: int = 2) -> dict:
    t = s["text"]
    n_full = sum(1 for x in t["layer_types"] if x == "full_attention")
    kv_per_token = n_full * 2 * t["num_kv_heads"] * t["head_dim"] * dtype_bytes
    gdn_state = 0
    if t["gdn"]:
        g = t["gdn"]
        n_lin = t["num_layers"] - n_full
        conv_dim = 2 * g["num_k_heads"] * g["k_head_dim"] + g["num_v_heads"] * g["v_head_dim"]
        # recurrent state S: [H_v, d_k, d_v] in fp32 + conv state [conv_dim, kernel-1]
        gdn_state = n_lin * (g["num_v_heads"] * g["k_head_dim"] * g["v_head_dim"] * 4 + conv_dim * (g["conv_kernel"] - 1) * dtype_bytes)
    return {"kv_cache_bytes_per_token": kv_per_token, "gdn_state_bytes_fixed": gdn_state}


def vision_tokens(s: dict, height: int, width: int, frames: int = 1) -> dict:
    v = s["vision"]
    unit = v["patch_size"] * v["spatial_merge_size"]
    h, w = max(unit, round(height / unit) * unit), max(unit, round(width / unit) * unit)
    gh, gw = h // v["patch_size"], w // v["patch_size"]
    gt = max(1, frames // v["temporal_patch_size"])
    return {"resized": [h, w], "grid_thw": [gt, gh, gw], "patches": gt * gh * gw,
            "llm_tokens": gt * gh * gw // v["spatial_merge_size"] ** 2}


def build(config_dir: Path = CONFIG_DIR) -> dict:
    out = []
    for m in MODELS:
        p = config_dir / m.key / "config.json"
        if not p.exists():
            continue
        cfg = json.loads(p.read_text())
        s = normalise(cfg, m.family)
        tp = text_params(s["text"])
        vp = vision_params(s["vision"]) if s["vision"] else {"total": 0}
        s["params"] = {"text": tp, "vision": vp,
                       "total": tp["total"] + vp["total"],
                       "active": tp["active"] + vp["total"],
                       "total_with_mtp": tp["total"] + vp["total"] + tp["mtp"]}
        s["costs"] = per_token_costs(s)
        s["costs"]["vision_tokens_448"] = vision_tokens(s, 448, 448) if s["vision"] else None
        s["costs"]["vision_tokens_1mp"] = vision_tokens(s, 1024, 1024) if s["vision"] else None
        pre = config_dir / m.key / "preprocessor_config.json"
        s["preprocessor"] = json.loads(pre.read_text()) if pre.exists() else None
        gen = config_dir / m.key / "generation_config.json"
        s["generation"] = json.loads(gen.read_text()) if gen.exists() else None
        out.append({**m.to_dict(), "spec": s})
    return {"models": out, "families": FAMILIES, "lineage": LINEAGE_NOTES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/specs/models.json")
    a = ap.parse_args()
    data = build()
    Path(a.out).write_text(json.dumps(data, indent=1))
    for m in data["models"]:
        p = m["spec"]["params"]
        print(f"{m['key']:22s} total {p['total']/1e9:7.2f}B  active {p['active']/1e9:6.2f}B  (+mtp {p['text']['mtp']/1e9:.2f}B)")


if __name__ == "__main__":
    main()
