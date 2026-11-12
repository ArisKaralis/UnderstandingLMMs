"""Config -> hierarchical architecture graph with symbolic + concrete tensor shapes.

Node levels:  macro (stage) > group (e.g. "LLM layers") > block (one layer) > op (one tensor op).
Every node carries: inputs/outputs shapes, op pseudo-code, LaTeX equation, params, FLOPs, prose
explanation and a reference. The same function is used for the real models, for config
overrides in the playground, and to validate captured traces (shapes must match).
"""
from __future__ import annotations

import copy
import math
from typing import Any

from lmmviz.specs.derive import full_attn_params, gdn_params, normalise, vision_tokens

Shape = list  # list of [symbol, value]


def S(*dims: tuple[str, int]) -> Shape:
    return [[s, int(v)] for s, v in dims]


class G:
    def __init__(self) -> None:
        self.nodes: list[dict] = []
        self.edges: list[dict] = []

    def add(self, id: str, label: str, kind: str, level: str, parent: str | None = None, *,
            inputs: dict[str, Shape] | None = None, outputs: dict[str, Shape] | None = None,
            params: int = 0, flops: int = 0, op: str = "", eq: str = "", info: str = "",
            ref: str = "", weights: dict[str, Shape] | None = None, meta: dict | None = None) -> str:
        self.nodes.append({
            "id": id, "label": label, "kind": kind, "level": level, "parent": parent,
            "inputs": inputs or {}, "outputs": outputs or {}, "weights": weights or {},
            "params": int(params), "flops": int(flops), "op": op.strip(), "eq": eq, "info": info,
            "ref": ref, "meta": meta or {},
        })
        return id

    def edge(self, a: str, b: str, label: str = "") -> None:
        self.edges.append({"source": a, "target": b, "label": label})


def mm_flops(m: int, k: int, n: int) -> int:
    return 2 * m * k * n


REF_VL = "Qwen3-VL Tech Report (arXiv 2511.21631)"
REF_GDN = "Gated Delta Networks (arXiv 2412.06464); Qwen3.8-Next report §2.1.1"
REF_GA = "Gated Attention for LLMs (arXiv 2505.06708)"


# --------------------------------------------------------------------------- vision
def vision_graph(g: G, spec: dict, img_hw: tuple[int, int]) -> dict:
    v, d = spec["vision"], spec["text"]["hidden_size"]
    vt = vision_tokens(spec, *img_hw)
    gt, gh, gw = vt["grid_thw"]
    Np, Nv = vt["patches"], vt["llm_tokens"]
    dv, fv, hv, hdv = v["hidden_size"], v["intermediate_size"], v["num_heads"], v["head_dim"]
    p, tp, m, C = v["patch_size"], v["temporal_patch_size"], v["spatial_merge_size"], v["in_channels"]
    patch_dim = C * tp * p * p
    H, W = vt["resized"]

    g.add("vision", "Vision encoder", "vision", "macro", None,
          inputs={"pixels": S(("C", C), ("H", H), ("W", W))}, outputs={"vision_embeds": S(("N_v", Nv), ("d", d))},
          info=f"SigLIP-2-style ViT ({v['depth']} blocks, width {dv}). The image is resized to multiples of "
               f"{p*m}px, cut into {p}×{p} patches ({gh}×{gw}={Np} patches), encoded, then every 2×2 group "
               f"is merged into one LLM token ({Nv} tokens).", ref=REF_VL)
    g.add("vision.preprocess", "Smart resize + patchify", "preprocess", "op", "vision",
          inputs={"image": S(("C", C), ("H₀", img_hw[0]), ("W₀", img_hw[1]))},
          outputs={"pixel_values": S(("N_p", Np), ("C·T·p·p", patch_dim)), "grid_thw": S(("3", 3))},
          op=f"""img = resize(img, to=({H},{W}))            # multiples of patch*merge = {p*m}
img = (img/255 - 0.5) / 0.5                    # mean=std=0.5
frames = repeat(img, T={tp})                   # a still image is 2 identical frames
patches = rearrange(frames, 'c (t pt) (h ph) (w pw) -> (t h w) (c pt ph pw)', pt={tp}, ph={p}, pw={p})
# patch order is grouped so each 2x2 merge window is contiguous""",
          eq=r"N_p = \tfrac{T}{%d}\cdot\tfrac{H}{%d}\cdot\tfrac{W}{%d}" % (tp, p, p),
          meta={"grid_thw": [gt, gh, gw], "resized": [H, W]},
          info="No learned weights. Dynamic resolution: the patch count follows the image size, bounded by min/max pixels.")
    pe_p = patch_dim * dv + dv
    g.add("vision.patch_embed", "Patch embedding (Conv3d)", "linear", "op", "vision",
          inputs={"pixel_values": S(("N_p", Np), ("C·T·p·p", patch_dim))}, outputs={"x": S(("N_p", Np), ("d_v", dv))},
          weights={"proj.weight": S(("d_v", dv), ("C", C), ("T", tp), ("p", p), ("p", p))},
          params=pe_p, flops=mm_flops(Np, patch_dim, dv),
          op=f"x = Conv3d(kernel=stride=({tp},{p},{p}))(patches)  # == patches @ W.T + b",
          eq=r"x_i = W_{pe}\,\mathrm{vec}(\text{patch}_i) + b", info="A stride-equal Conv3d is exactly a linear projection of each flattened 2×16×16×3 patch.")
    npos = v["num_position_embeddings"]
    side = int(math.isqrt(npos))
    g.add("vision.pos_embed", "Absolute pos-emb (bilinear interp.)", "embedding", "op", "vision",
          inputs={"x": S(("N_p", Np), ("d_v", dv))}, outputs={"x": S(("N_p", Np), ("d_v", dv))},
          weights={"pos_embed.weight": S(("P", npos), ("d_v", dv))}, params=npos * dv,
          op=f"grid = pos_embed.weight.view({side},{side},d_v)\npe = bilinear_interp(grid, to=({gh},{gw}))\nx = x + pe.flatten(0,1)",
          info=f"A learned {side}×{side} position grid is resampled to the image's {gh}×{gw} patch grid (CoMP-style), so any resolution works.")
    g.add("vision.rope2d", "2D RoPE (h, w)", "rope", "op", "vision",
          inputs={"pos_ids": S(("N_p", Np), ("2", 2))}, outputs={"cos,sin": S(("N_p", Np), ("d_h", hdv))},
          op="half of each head's rotary dims rotate with the row index, half with the column index",
          eq=r"\mathrm{RoPE}(q)_{[h\text{-dims}]}\!\propto\! e^{i\,\theta_j\,\text{row}},\;\mathrm{RoPE}(q)_{[w\text{-dims}]}\!\propto\! e^{i\,\theta_j\,\text{col}}")
    blocks = g.add("vision.blocks", f"ViT blocks ×{v['depth']}", "group", "group", "vision",
                   inputs={"x": S(("N_p", Np), ("d_v", dv))}, outputs={"x": S(("N_p", Np), ("d_v", dv))},
                   params=v["depth"] * ((dv * 3 * dv + 3 * dv) + (dv * dv + dv) + (dv * fv + fv) + (fv * dv + dv) + 4 * dv))
    ds = v["deepstack_visual_indexes"]
    for i in range(v["depth"]):
        b = f"vision.block.{i}"
        attn_p = dv * 3 * dv + 3 * dv + dv * dv + dv
        mlp_p = dv * fv + fv + fv * dv + dv
        fl = mm_flops(Np, dv, 3 * dv) + 4 * Np * Np * dv + mm_flops(Np, dv, dv) + 2 * mm_flops(Np, dv, fv)
        g.add(b, f"ViT block {i}" + ("  ⟶ DeepStack tap" if i in ds else ""), "vit_block", "block", blocks,
              inputs={"x": S(("N_p", Np), ("d_v", dv))}, outputs={"x": S(("N_p", Np), ("d_v", dv))},
              params=attn_p + mlp_p + 4 * dv, flops=fl, meta={"index": i, "deepstack_tap": i in ds},
              eq=r"x \leftarrow x + \mathrm{Attn}(\mathrm{LN}(x));\quad x \leftarrow x + \mathrm{MLP}(\mathrm{LN}(x))")
        g.add(f"{b}.norm1", "LayerNorm", "norm", "op", b, inputs={"x": S(("N_p", Np), ("d_v", dv))},
              outputs={"x̂": S(("N_p", Np), ("d_v", dv))}, params=2 * dv, op="x̂ = (x-μ)/σ · γ + β")
        g.add(f"{b}.attn.qkv", "QKV projection", "linear", "op", b,
              inputs={"x̂": S(("N_p", Np), ("d_v", dv))},
              outputs={"q,k,v": S(("3", 3), ("H", hv), ("N_p", Np), ("d_h", hdv))},
              weights={"qkv.weight": S(("3·d_v", 3 * dv), ("d_v", dv))}, params=dv * 3 * dv + 3 * dv,
              flops=mm_flops(Np, dv, 3 * dv), op="q, k, v = (x̂ @ W_qkv.T + b).view(N_p, 3, H, d_h).unbind(1)")
        g.add(f"{b}.attn.sdpa", "Bidirectional attention", "attention", "op", b,
              inputs={"q,k,v": S(("H", hv), ("N_p", Np), ("d_h", hdv))},
              outputs={"o": S(("N_p", Np), ("d_v", dv))}, flops=4 * Np * Np * dv,
              op="q, k = rope2d(q), rope2d(k)\nscores = einsum('hqd,hkd->hqk', q, k) / sqrt(d_h)   # [H, N_p, N_p]\np = softmax(scores, -1)   # no causal mask; masked per image via cu_seqlens\no = einsum('hqk,hkd->qhd', p, v).reshape(N_p, d_v)",
              eq=r"\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right)V",
              meta={"score_shape": [hv, Np, Np]})
        g.add(f"{b}.attn.proj", "Output projection", "linear", "op", b, inputs={"o": S(("N_p", Np), ("d_v", dv))},
              outputs={"Δx": S(("N_p", Np), ("d_v", dv))}, weights={"proj.weight": S(("d_v", dv), ("d_v", dv))},
              params=dv * dv + dv, flops=mm_flops(Np, dv, dv), op="x = x + o @ W_o.T + b")
        g.add(f"{b}.norm2", "LayerNorm", "norm", "op", b, inputs={"x": S(("N_p", Np), ("d_v", dv))},
              outputs={"x̂": S(("N_p", Np), ("d_v", dv))}, params=2 * dv)
        g.add(f"{b}.mlp", f"MLP ({v['hidden_act']})", "mlp", "op", b, inputs={"x̂": S(("N_p", Np), ("d_v", dv))},
              outputs={"Δx": S(("N_p", Np), ("d_v", dv))},
              weights={"fc1": S(("d_ff", fv), ("d_v", dv)), "fc2": S(("d_v", dv), ("d_ff", fv))},
              params=mlp_p, flops=2 * mm_flops(Np, dv, fv), op=f"x = x + fc2(gelu_tanh(fc1(x̂)))   # {dv} → {fv} → {dv}")
    m4 = dv * m * m
    merger_p = 2 * dv + m4 * m4 + m4 + m4 * d + d
    g.add("vision.merger", f"Patch merger {m}×{m} → MLP", "merger", "op", "vision",
          inputs={"x": S(("N_p", Np), ("d_v", dv))}, outputs={"vision_embeds": S(("N_v", Nv), ("d", d))},
          weights={"fc1": S(("4d_v", m4), ("4d_v", m4)), "fc2": S(("d", d), ("4d_v", m4))},
          params=merger_p, flops=mm_flops(Nv, m4, m4) + mm_flops(Nv, m4, d),
          op=f"x = LayerNorm(x)\nx = x.view(N_v, {m*m}*d_v)        # 2×2 neighbouring patches concatenated → [{Nv}, {m4}]\nx = fc2(gelu(fc1(x)))            # {m4} → {m4} → {d} (LLM width)",
          eq=r"e_j = W_2\,\mathrm{GELU}(W_1[\,x_{j,1};x_{j,2};x_{j,3};x_{j,4}\,])",
          info="Compresses 4 patches into one visual token at the LLM hidden size: this is where vision 'becomes' language-model input.")
    for k, idx in enumerate(ds):
        g.add(f"vision.deepstack.{k}", f"DeepStack merger {k} (from ViT layer {idx})", "merger", "op", "vision",
              inputs={"x": S(("N_p", Np), ("d_v", dv))}, outputs={"ds_embeds": S(("N_v", Nv), ("d", d))},
              params=2 * m4 + m4 * m4 + m4 + m4 * d + d,
              op=f"ds_{k} = Merger_{k}(ViT_block_{idx}.output)   # post-shuffle LayerNorm over {m4}\n# later added to LLM layer {k} hidden states at the image positions",
              meta={"vit_layer": idx, "llm_layer": k}, ref=REF_VL,
              info="DeepStack: features from intermediate ViT layers carry fine detail that the last layer loses; they are injected into the first LLM layers without adding tokens.")
    g.edge("vision.preprocess", "vision.patch_embed"); g.edge("vision.patch_embed", "vision.pos_embed")
    g.edge("vision.pos_embed", "vision.blocks"); g.edge("vision.blocks", "vision.merger")
    return {"Np": Np, "Nv": Nv, "grid_thw": [gt, gh, gw], "resized": [H, W]}


# --------------------------------------------------------------------------- text layers
def attn_layer_ops(g: G, b: str, t: dict, T: int, ctx_len: int) -> None:
    d, hq, hkv, hd, rd = t["hidden_size"], t["num_q_heads"], t["num_kv_heads"], t["head_dim"], t["rotary_dim"]
    gate = t["attn_output_gate"]
    qo = hq * hd * (2 if gate else 1)
    g.add(f"{b}.q_proj", "Q projection" + (" (+ gate)" if gate else ""), "linear", "op", b,
          inputs={"x̂": S(("T", T), ("d", d))},
          outputs={"q": S(("T", T), ("H_q", hq), ("d_h", hd))} | ({"gate": S(("T", T), ("H_q·d_h", hq * hd))} if gate else {}),
          weights={"q_proj.weight": S(("H_q·d_h" + ("·2" if gate else ""), qo), ("d", d))},
          params=d * qo, flops=mm_flops(T, d, qo),
          op=("q, gate = (x̂ @ W_q.T).view(T, H_q, 2*d_h).chunk(2, -1)  # gate packed in q_proj" if gate
              else "q = (x̂ @ W_q.T).view(T, H_q, d_h)"))
    g.add(f"{b}.kv_proj", "K, V projections (GQA)", "linear", "op", b, inputs={"x̂": S(("T", T), ("d", d))},
          outputs={"k": S(("T", T), ("H_kv", hkv), ("d_h", hd)), "v": S(("T", T), ("H_kv", hkv), ("d_h", hd))},
          weights={"k_proj.weight": S(("H_kv·d_h", hkv * hd), ("d", d)), "v_proj.weight": S(("H_kv·d_h", hkv * hd), ("d", d))},
          params=2 * d * hkv * hd, flops=2 * mm_flops(T, d, hkv * hd),
          op=f"k = (x̂ @ W_k.T).view(T, H_kv, d_h)\nv = (x̂ @ W_v.T).view(T, H_kv, d_h)\n# {hq // hkv} query heads share each KV head",
          info=f"Grouped-query attention: only {hkv} K/V heads are stored in the KV cache, shared by {hq} query heads.")
    g.add(f"{b}.qk_norm", "QK-RMSNorm (per head)", "norm", "op", b,
          inputs={"q": S(("T", T), ("H_q", hq), ("d_h", hd))}, outputs={"q,k": S(("T", T), ("H", hq), ("d_h", hd))},
          params=2 * hd, op="q = RMSNorm_dh(q); k = RMSNorm_dh(k)   # stabilises attention logits")
    rope_txt = (f"apply rotary to the first {rd} of {hd} dims (partial rotary {rd/hd:.2f}); rest pass through"
                if rd < hd else f"rotary over all {hd} dims")
    sec = t["rope"]["mrope_section"]
    g.add(f"{b}.rope", "Interleaved MRoPE", "rope", "op", b,
          inputs={"q,k": S(("T", T), ("H", hq), ("d_h", hd)), "pos_ids": S(("3", 3), ("T", T))},
          outputs={"q,k": S(("T", T), ("H", hq), ("d_h", hd))},
          op=f"{rope_txt}\n# frequency j uses axis: H if j%3==1 and j<3*{sec[1]}, W if j%3==2 and j<3*{sec[2]}, else T\n# mrope_section={sec} (pairs per t/h/w), θ={t['rope']['theta']:.0e}",
          eq=r"q'_{2j:2j+2} = R(\theta_j\, p_{a(j)})\, q_{2j:2j+2},\quad a(j)\in\{t,h,w\}",
          info="Each token has a 3-D position (t,h,w). Text tokens use t=h=w. Interleaving spreads every axis across low and high frequencies.",
          ref=REF_VL, meta={"mrope_section": sec, "rotary_dim": rd})
    kv_len = T
    g.add(f"{b}.sdpa", "Causal attention", "attention", "op", b,
          inputs={"q": S(("T", T), ("H_q", hq), ("d_h", hd)), "k,v": S(("T_kv", kv_len), ("H_kv", hkv), ("d_h", hd))},
          outputs={"o": S(("T", T), ("H_q", hq), ("d_h", hd))}, flops=2 * T * kv_len * hq * hd,
          op="k, v = repeat_kv(k, v, H_q // H_kv)\nscores = einsum('qhd,khd->hqk', q, k) * d_h**-0.5   # [H_q, T, T_kv]\nscores += causal_mask\np = softmax(scores.float(), -1)\no = einsum('hqk,khd->qhd', p, v)",
          eq=r"o = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M_{\text{causal}}\right)V",
          meta={"score_shape": [hq, T, kv_len], "kv_cache_bytes_per_token_layer": 2 * hkv * hd * 2})
    if gate:
        g.add(f"{b}.gate", "Sigmoid output gate", "gate", "op", b,
              inputs={"o": S(("T", T), ("H_q·d_h", hq * hd)), "gate": S(("T", T), ("H_q·d_h", hq * hd))},
              outputs={"o": S(("T", T), ("H_q·d_h", hq * hd))}, flops=2 * T * hq * hd,
              op="o = o * sigmoid(gate)   # elementwise, head-specific, query-dependent",
              eq=r"Y' = \mathrm{SDPA}(Q,K,V)\odot\sigma(XW_\theta)", ref=REF_GA,
              info="The gate adds non-linearity between W_V and W_O and lets heads output ~0, removing the 'attention sink' on the first token.")
    g.add(f"{b}.o_proj", "Output projection", "linear", "op", b, inputs={"o": S(("T", T), ("H_q·d_h", hq * hd))},
          outputs={"Δh": S(("T", T), ("d", d))}, weights={"o_proj.weight": S(("d", d), ("H_q·d_h", hq * hd))},
          params=hq * hd * d, flops=mm_flops(T, hq * hd, d), op="Δh = o.reshape(T, H_q*d_h) @ W_o.T")


def gdn_layer_ops(g: G, b: str, t: dict, T: int) -> None:
    d, gd = t["hidden_size"], t["gdn"]
    hk, hv, dk, dvh, kk = gd["num_k_heads"], gd["num_v_heads"], gd["k_head_dim"], gd["v_head_dim"], gd["conv_kernel"]
    key_dim, val_dim = hk * dk, hv * dvh
    conv_dim = 2 * key_dim + val_dim
    g.add(f"{b}.in_proj", "In-projections (q,k,v,z,α,β)", "linear", "op", b,
          inputs={"x̂": S(("T", T), ("d", d))},
          outputs={"qkv": S(("T", T), ("2·H_k·d_k+H_v·d_v", conv_dim)), "z": S(("T", T), ("H_v", hv), ("d_v", dvh)),
                   "a,b": S(("T", T), ("H_v", hv))},
          weights={"in_proj_qkv": S(("conv_dim", conv_dim), ("d", d)), "in_proj_z": S(("H_v·d_v", val_dim), ("d", d)),
                   "in_proj_a": S(("H_v", hv), ("d", d)), "in_proj_b": S(("H_v", hv), ("d", d))},
          params=d * conv_dim + d * val_dim + 2 * d * hv, flops=mm_flops(T, d, conv_dim + val_dim + 2 * hv),
          op="qkv = x̂ @ W_qkv.T;  z = x̂ @ W_z.T;  a = x̂ @ W_a.T;  b = x̂ @ W_b.T")
    g.add(f"{b}.conv", f"Short causal conv (k={kk}) + SiLU", "conv", "op", b,
          inputs={"qkv": S(("conv_dim", conv_dim), ("T", T))}, outputs={"qkv": S(("conv_dim", conv_dim), ("T", T))},
          weights={"conv1d.weight": S(("conv_dim", conv_dim), ("1", 1), ("k", kk))}, params=conv_dim * kk,
          flops=2 * conv_dim * kk * T,
          op=f"qkv = silu(depthwise_conv1d(qkv, kernel={kk}, causal=True))\nq, k, v = split(qkv, [{key_dim}, {key_dim}, {val_dim}])",
          info="Each channel mixes only the last 4 tokens: a cheap local inductive bias before compressing into the state.")
    g.add(f"{b}.gates", "Decay α and write strength β", "gate", "op", b,
          inputs={"a,b": S(("T", T), ("H_v", hv))}, outputs={"g=log α": S(("T", T), ("H_v", hv)), "β": S(("T", T), ("H_v", hv))},
          weights={"A_log": S(("H_v", hv),), "dt_bias": S(("H_v", hv),)}, params=2 * hv,
          op="beta = sigmoid(b)                                # how strongly to write\ng = -exp(A_log) * softplus(a + dt_bias)            # log of decay α ∈ (0,1)",
          eq=r"\beta_t=\sigma(W_\beta x_t),\quad \alpha_t=\exp\!\big(-e^{A}\,\mathrm{softplus}(W_\alpha x_t+b)\big)", ref=REF_GDN)
    g.add(f"{b}.delta", "Gated delta rule (recurrent state)", "recurrence", "op", b,
          inputs={"q,k": S(("T", T), ("H_v", hv), ("d_k", dk)), "v": S(("T", T), ("H_v", hv), ("d_v", dvh))},
          outputs={"y": S(("T", T), ("H_v", hv), ("d_v", dvh)), "state S": S(("H_v", hv), ("d_k", dk), ("d_v", dvh))},
          flops=T * hv * dk * dvh * 6,
          op=f"q, k = l2norm(q), l2norm(k)       # q,k heads repeated {hv // hk}× to match H_v\nfor t in range(T):                  # (chunked WY form in practice, chunk=64)\n    S = α_t * S                                     # forget\n    e = v_t - S.T @ k_t                             # what the memory gets wrong for k_t\n    S = S + β_t * outer(k_t, e)                     # write only the error (delta rule)\n    y_t = S.T @ q_t",
          eq=r"S_t=(\alpha_t I-\beta_t k_t k_t^\top)S_{t-1}+\beta_t k_t v_t^\top,\qquad y_t=S_t^\top q_t",
          meta={"state_shape": [hv, dk, dvh], "state_bytes_fp32": hv * dk * dvh * 4},
          info=f"Linear attention with a fixed-size memory: per head a {dk}×{dvh} matrix, independent of sequence length (no KV cache).",
          ref=REF_GDN)
    g.add(f"{b}.out", "Gated RMSNorm + out projection", "linear", "op", b,
          inputs={"y": S(("T", T), ("H_v", hv), ("d_v", dvh)), "z": S(("T", T), ("H_v", hv), ("d_v", dvh))},
          outputs={"Δh": S(("T", T), ("d", d))},
          weights={"norm.weight": S(("d_v", dvh),), "out_proj.weight": S(("d", d), ("H_v·d_v", val_dim))},
          params=dvh + val_dim * d, flops=mm_flops(T, val_dim, d),
          op="y = RMSNorm(y) * silu(z)          # per head\nΔh = y.reshape(T, H_v*d_v) @ W_out.T",
          eq=r"o_t = W_o\big[\mathrm{RMSNorm}(y_t)\odot \mathrm{SiLU}(W_z x_t)\big]")


def ffn_ops(g: G, b: str, t: dict, T: int) -> None:
    d = t["hidden_size"]
    if not t["moe"]:
        f = t["intermediate_size"]
        g.add(f"{b}.mlp", "SwiGLU MLP", "mlp", "op", b, inputs={"x̂": S(("T", T), ("d", d))}, outputs={"Δh": S(("T", T), ("d", d))},
              weights={"gate_proj": S(("d_ff", f), ("d", d)), "up_proj": S(("d_ff", f), ("d", d)), "down_proj": S(("d", d), ("d_ff", f))},
              params=3 * d * f, flops=3 * mm_flops(T, d, f),
              op=f"Δh = down(silu(gate(x̂)) * up(x̂))    # {d} → {f} → {d}", eq=r"W_d\big(\mathrm{SiLU}(W_g x)\odot W_u x\big)")
        return
    m = t["moe"]
    E, k, fe, fs = m["num_experts"], m["top_k"], m["expert_ffn"], m["shared_ffn"]
    g.add(f"{b}.router", f"Router (top-{k} of {E})", "router", "op", b, inputs={"x̂": S(("T", T), ("d", d))},
          outputs={"weights": S(("T", T), ("k", k)), "experts": S(("T", T), ("k", k))},
          weights={"gate.weight": S(("E", E), ("d", d))}, params=E * d, flops=mm_flops(T, d, E),
          op="logits = x̂ @ W_r.T                 # [T, E]\np = softmax(logits)\nw, idx = topk(p, k)\nw = w / w.sum(-1, keepdim=True)    # renormalise over the chosen experts",
          eq=r"\mathcal{E}(x)=\mathrm{TopK}_k(\mathrm{softmax}(W_r x)),\; w_e=\frac{p_e}{\sum_{e'\in\mathcal{E}}p_{e'}}")
    g.add(f"{b}.experts", f"{E} SwiGLU experts (each d_ff={fe})", "experts", "op", b,
          inputs={"x̂": S(("T", T), ("d", d))}, outputs={"Δh": S(("T", T), ("d", d))},
          weights={"gate_up_proj": S(("E", E), ("2·d_ff", 2 * fe), ("d", d)), "down_proj": S(("E", E), ("d", d), ("d_ff", fe))},
          params=E * 3 * d * fe, flops=k * 3 * mm_flops(T, d, fe),
          op=f"out = Σ_(e in idx) w_e · Expert_e(x̂)     # only {k}/{E} experts run per token",
          meta={"active_params_per_token": k * 3 * d * fe})
    if fs:
        g.add(f"{b}.shared", "Shared expert (sigmoid-gated)", "mlp", "op", b, inputs={"x̂": S(("T", T), ("d", d))},
              outputs={"Δh": S(("T", T), ("d", d))}, params=3 * d * fs + d, flops=3 * mm_flops(T, d, fs),
              op="out += sigmoid(x̂ @ w_sg) * SharedExpert(x̂)   # always active")


def build_graph(spec: dict, n_text: int = 48, img_hw: tuple[int, int] | None = (448, 448),
                gen_tokens: int = 1) -> dict:
    g = G()
    t = spec["text"]
    d, V, L = t["hidden_size"], t["vocab_size"], t["num_layers"]
    vis = None
    if spec.get("vision") and img_hw:
        vis = vision_graph(g, spec, img_hw)
    Nv = vis["Nv"] if vis else 0
    T = n_text + (Nv + 2 if vis else 0)

    g.add("input", "Input (chat template)", "input", "macro", None,
          outputs={"input_ids": S(("T", T),)},
          info=f"Prompt rendered with the chat template: {n_text} text tokens" +
               (f" + <|vision_start|> + {Nv} <|image_pad|> + <|vision_end|>" if vis else ""),
          meta={"n_text": n_text, "n_vision": Nv, "T": T})
    g.add("embed", "Token embedding + vision scatter", "embedding", "macro", None,
          inputs={"input_ids": S(("T", T),)} | ({"vision_embeds": S(("N_v", Nv), ("d", d))} if vis else {}),
          outputs={"h₀": S(("T", T), ("d", d))}, weights={"embed_tokens": S(("V", V), ("d", d))}, params=V * d,
          op="h = embed_tokens[input_ids]          # [T, d]\nh[input_ids == <|image_pad|>] = vision_embeds   # early fusion: same sequence",
          info="Image placeholder rows are overwritten by the merger outputs, so text and image tokens share one sequence.")
    g.add("positions", "3-D position ids (t,h,w)", "rope", "macro", None,
          inputs={"input_ids": S(("T", T),), "grid_thw": S(("3", 3),)}, outputs={"position_ids": S(("3", 3), ("T", T))},
          op="text: t=h=w=running index\nimage: t=const, h=row, w=col (offset by preceding text); next text resumes at max+1",
          ref=REF_VL)
    llm = g.add("llm", f"Decoder layers ×{L}", "group", "macro", None,
                inputs={"h": S(("T", T), ("d", d))}, outputs={"h": S(("T", T), ("d", d))})
    attn_p, gdn_p = full_attn_params(t), (gdn_params(t) if t["gdn"] else 0)
    ds_layers = len(spec["vision"]["deepstack_visual_indexes"]) if spec.get("vision") else 0
    for i, lt in enumerate(t["layer_types"]):
        b = f"llm.layer.{i}"
        mixer_p = attn_p if lt == "full_attention" else gdn_p
        ffn_kind = "moe" if t["moe"] else "dense"
        g.add(b, f"Layer {i} · {'Gated DeltaNet' if lt == 'linear_attention' else ('Gated attention' if t['attn_output_gate'] else 'Attention')} · {'MoE' if t['moe'] else 'MLP'}",
              "decoder_layer", "block", llm, inputs={"h": S(("T", T), ("d", d))}, outputs={"h": S(("T", T), ("d", d))},
              params=mixer_p, meta={"index": i, "mixer": lt, "ffn": ffn_kind, "deepstack": i < ds_layers and vis is not None},
              eq=r"h \leftarrow h + \mathrm{Mixer}(\mathrm{RMSNorm}(h));\quad h \leftarrow h + \mathrm{FFN}(\mathrm{RMSNorm}(h))")
        g.add(f"{b}.input_norm", "RMSNorm", "norm", "op", b, inputs={"h": S(("T", T), ("d", d))},
              outputs={"x̂": S(("T", T), ("d", d))}, params=d,
              op="x̂ = h / rms(h) * (1 + w)" if t["gdn"] else "x̂ = h / rms(h) * w",
              info="Qwen3.5 uses zero-centred RMSNorm weights (scale = 1 + w)." if t["gdn"] else "")
        if lt == "full_attention":
            attn_layer_ops(g, b, t, T, T)
        else:
            gdn_layer_ops(g, b, t, T)
        g.add(f"{b}.residual1", "Residual add", "add", "op", b, inputs={"h": S(("T", T), ("d", d)), "Δh": S(("T", T), ("d", d))},
              outputs={"h": S(("T", T), ("d", d))}, op="h = h + Δh")
        g.add(f"{b}.post_norm", "RMSNorm", "norm", "op", b, inputs={"h": S(("T", T), ("d", d))}, outputs={"x̂": S(("T", T), ("d", d))}, params=d)
        ffn_ops(g, b, t, T)
        g.add(f"{b}.residual2", "Residual add", "add", "op", b, outputs={"h": S(("T", T), ("d", d))}, op="h = h + Δh")
        if i < ds_layers and vis is not None:
            g.add(f"{b}.deepstack_add", f"+ DeepStack features {i}", "add", "op", b,
                  inputs={"h[vision]": S(("N_v", Nv), ("d", d)), "ds": S(("N_v", Nv), ("d", d))},
                  outputs={"h": S(("T", T), ("d", d))}, op=f"h[visual_pos_mask] += ds_{i}   # after layer {i}", ref=REF_VL)
    g.add("final_norm", "Final RMSNorm", "norm", "macro", None, inputs={"h": S(("T", T), ("d", d))}, outputs={"h": S(("T", T), ("d", d))}, params=d)
    g.add("lm_head", "LM head" + (" (tied to embedding)" if t["tie_word_embeddings"] else ""), "linear", "macro", None,
          inputs={"h_last": S(("1", 1), ("d", d))}, outputs={"logits": S(("1", 1), ("V", V))},
          weights={"lm_head.weight": S(("V", V), ("d", d))}, params=0 if t["tie_word_embeddings"] else V * d,
          flops=mm_flops(1, d, V), op="logits = h[-1] @ W_E.T  # only the last position is needed while generating\np = softmax(logits / temperature) → top-k/top-p → sample")
    if t["mtp_layers"]:
        g.add("mtp", "Multi-token prediction head", "mtp", "macro", None,
              inputs={"h": S(("T", T), ("d", d)), "emb(t+1)": S(("T", T), ("d", d))}, outputs={"logits(t+2)": S(("T", T), ("V", V))},
              op="h' = W_fc [RMSNorm(h); RMSNorm(Emb(x_{t+1}))]   # 2d → d\nh' = DecoderLayer_full_attn(h')\nlogits = LMHead(RMSNorm(h'))     # draft for speculative decoding",
              eq=r"h'=M\,[\mathrm{RMSNorm}(h);\mathrm{RMSNorm}(\mathrm{Emb}(t_{i+1}))]",
              info="Trained to predict one token further ahead. At inference its drafts are verified by the main model (speculative decoding).",
              ref="DeepSeek-V3 §2.2 (arXiv 2412.19437)")
    # macro edges
    if vis:
        g.edge("vision", "embed", "vision_embeds")
    g.edge("input", "embed"); g.edge("input", "positions"); g.edge("embed", "llm"); g.edge("positions", "llm", "cos/sin")
    g.edge("llm", "final_norm"); g.edge("final_norm", "lm_head")
    if t["mtp_layers"]:
        g.edge("final_norm", "mtp")
    for k in range(ds_layers if vis else 0):
        g.edge(f"vision.deepstack.{k}", f"llm.layer.{k}", "DeepStack")
    total_flops = sum(n["flops"] for n in g.nodes if n["level"] == "op")
    return {"nodes": g.nodes, "edges": g.edges,
            "context": {"T": T, "n_text": n_text, "n_vision": Nv, "vision": vis, "img_hw": list(img_hw) if img_hw else None},
            "totals": {"prefill_flops": total_flops}}


def apply_overrides(spec: dict, overrides: dict[str, Any]) -> dict:
    """Playground: change config fields, e.g. {"text.num_layers": 12, "vision.patch_size": 14}."""
    s = copy.deepcopy(spec)
    for path, val in overrides.items():
        cur = s
        parts = path.split(".")
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = val
    t = s["text"]
    # keep layer_types consistent with num_layers / attention interval
    if "text.num_layers" in overrides or "text.full_attention_interval" in overrides:
        L = t["num_layers"]
        interval = overrides.get("text.full_attention_interval") or (4 if t["gdn"] else 1)
        if t["gdn"]:
            t["layer_types"] = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(L)]
        else:
            t["layer_types"] = ["full_attention"] * L
    if "text.head_dim" in overrides or "text.rope.partial_rotary_factor" in overrides:
        t["rotary_dim"] = int(t["head_dim"] * t["rope"]["partial_rotary_factor"])
    if s.get("vision"):
        s["vision"]["head_dim"] = s["vision"]["hidden_size"] // s["vision"]["num_heads"]
        s["vision"]["out_hidden_size"] = t["hidden_size"]
    return s


__all__ = ["build_graph", "apply_overrides", "normalise"]
