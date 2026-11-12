"""Forward hooks that record what each part of a Qwen-VL model does.

Two kinds of recording:
* generic: stats (and optionally full tensors) for every mapped module output;
* special: things a plain output hook cannot see, recomputed from the module inputs
  (attention probabilities under SDPA, gated-attention gate values, Gated DeltaNet
  decay/write gates and recurrent-state norms, MoE routing, MRoPE position ids, logit lens).
"""
from __future__ import annotations

import math
import re
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from .policy import Policy, selected_layers
from .stats import tensor_stats

# module name (regex) -> graph node id template
NODE_MAP: list[tuple[str, str]] = [
    (r"^model\.visual\.patch_embed$", "vision.patch_embed"),
    (r"^model\.visual\.blocks\.(\d+)$", "vision.block.{0}"),
    (r"^model\.visual\.blocks\.(\d+)\.norm1$", "vision.block.{0}.norm1"),
    (r"^model\.visual\.blocks\.(\d+)\.attn\.qkv$", "vision.block.{0}.attn.qkv"),
    (r"^model\.visual\.blocks\.(\d+)\.attn\.proj$", "vision.block.{0}.attn.proj"),
    (r"^model\.visual\.blocks\.(\d+)\.norm2$", "vision.block.{0}.norm2"),
    (r"^model\.visual\.blocks\.(\d+)\.mlp$", "vision.block.{0}.mlp"),
    (r"^model\.visual\.merger$", "vision.merger"),
    (r"^model\.visual\.deepstack_merger_list\.(\d+)$", "vision.deepstack.{0}"),
    (r"^model\.language_model\.embed_tokens$", "embed"),
    (r"^model\.language_model\.layers\.(\d+)$", "llm.layer.{0}"),
    (r"^model\.language_model\.layers\.(\d+)\.input_layernorm$", "llm.layer.{0}.input_norm"),
    (r"^model\.language_model\.layers\.(\d+)\.post_attention_layernorm$", "llm.layer.{0}.post_norm"),
    (r"^model\.language_model\.layers\.(\d+)\.self_attn\.q_proj$", "llm.layer.{0}.q_proj"),
    (r"^model\.language_model\.layers\.(\d+)\.self_attn\.[kv]_proj$", "llm.layer.{0}.kv_proj"),
    (r"^model\.language_model\.layers\.(\d+)\.self_attn\.[qk]_norm$", "llm.layer.{0}.qk_norm"),
    (r"^model\.language_model\.layers\.(\d+)\.self_attn\.o_proj$", "llm.layer.{0}.o_proj"),
    (r"^model\.language_model\.layers\.(\d+)\.linear_attn\.in_proj_(qkv|z|a|b)$", "llm.layer.{0}.in_proj"),
    (r"^model\.language_model\.layers\.(\d+)\.linear_attn\.norm$", "llm.layer.{0}.out"),
    (r"^model\.language_model\.layers\.(\d+)\.linear_attn\.out_proj$", "llm.layer.{0}.out"),
    (r"^model\.language_model\.layers\.(\d+)\.mlp$", "llm.layer.{0}.mlp"),
    (r"^model\.language_model\.layers\.(\d+)\.mlp\.gate$", "llm.layer.{0}.router"),
    (r"^model\.language_model\.layers\.(\d+)\.mlp\.shared_expert$", "llm.layer.{0}.shared"),
    (r"^model\.language_model\.norm$", "final_norm"),
    (r"^lm_head$", "lm_head"),
]
_COMPILED = [(re.compile(p), t) for p, t in NODE_MAP]


def node_for(name: str) -> str | None:
    for rx, tmpl in _COMPILED:
        m = rx.match(name)
        if m:
            return tmpl.format(*m.groups())
    return None


def _first_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for o in out:
            if isinstance(o, torch.Tensor):
                return o
    for attr in ("pooler_output", "last_hidden_state"):
        if hasattr(out, attr) and isinstance(getattr(out, attr), torch.Tensor):
            return getattr(out, attr)
    return None


def _modeling_module(mod):
    return sys.modules[type(mod).__module__]


class Recorder:
    """Attach to a model, run one teacher-forced forward pass, collect everything."""

    def __init__(self, model, policy: Policy, info: dict):
        self.model, self.policy, self.info = model, policy, info
        self.stats: list[dict] = []
        self.tensors: dict[str, torch.Tensor] = {}
        self.previews: dict[str, np.ndarray] = {}
        self.handles = []
        self.busy = False  # True while a special hook recomputes sub-modules (skip generic recording)
        self.layer_out: dict[int, torch.Tensor] = {}
        self.vit_out: dict[int, torch.Tensor] = {}
        self.cache: dict = defaultdict(dict)
        tc = model.config.get_text_config()
        self.n_layers = tc.num_hidden_layers
        vc = getattr(model.config, "vision_config", None)
        self.vit_depth = vc.depth if vc is not None else 0
        self.ds_idx = list(getattr(vc, "deepstack_visual_indexes", []) or []) if vc is not None else []
        self.sel_layers = selected_layers(self.n_layers)
        self.sel_vit = selected_layers(self.vit_depth) | set(self.ds_idx)

    # ------------------------------------------------------------------ attach
    def attach(self) -> "Recorder":
        for name, mod in self.model.named_modules():
            node = node_for(name)
            if node is not None:
                self.handles.append(mod.register_forward_hook(self._generic(name, node)))
            if name.endswith("language_model.rotary_emb"):
                self.handles.append(mod.register_forward_hook(self._rope_ids))
            m = re.match(r"^model\.visual\.blocks\.(\d+)\.attn$", name)
            if m and self._want_vit_attn(int(m.group(1))):
                self.handles.append(mod.register_forward_hook(self._vit_attn(int(m.group(1))), with_kwargs=True))
            m = re.match(r"^model\.language_model\.layers\.(\d+)\.self_attn$", name)
            if m and self._want_text_attn(int(m.group(1))):
                self.handles.append(mod.register_forward_hook(self._text_attn(int(m.group(1))), with_kwargs=True))
            m = re.match(r"^model\.language_model\.layers\.(\d+)\.linear_attn$", name)
            if m:
                self.handles.append(mod.register_forward_hook(self._gdn(int(m.group(1))), with_kwargs=True))
            m = re.match(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate$", name)
            if m:
                self.handles.append(mod.register_forward_hook(self._router(int(m.group(1)))))
        return self

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _want_vit_attn(self, i: int) -> bool:
        return self.policy.vision_attn_layers == "all" or i in self.sel_vit or i % 4 == 3

    def _want_text_attn(self, i: int) -> bool:
        return self.policy.text_attn_layers == "all" or i in self.sel_layers or i % 4 == 3

    def _keep_full(self, node: str) -> bool:
        if self.policy.full_tensors == "all":
            return True
        if self.policy.full_tensors == "none":
            return False
        m = re.match(r"llm\.layer\.(\d+)$", node)
        if m:
            return int(m.group(1)) in self.sel_layers
        m = re.match(r"vision\.block\.(\d+)$", node)
        if m:
            return int(m.group(1)) in self.sel_vit
        return node in ("embed", "vision.patch_embed", "vision.merger", "final_norm") or node.startswith("vision.deepstack")

    # ------------------------------------------------------------------ generic
    def _generic(self, name: str, node: str):
        def hook(mod, args, out):
            if self.busy:
                return
            t = _first_tensor(out)
            if t is None:
                return
            if node == "lm_head":  # [1,T,V] is big: keep stats only
                self.stats.append(tensor_stats(name, node, t[..., :4096], self.policy.hist_bins) | {"note": "first 4096 vocab cols"})
                return
            self.stats.append(tensor_stats(name, node, t, self.policy.hist_bins))
            m = re.match(r"llm\.layer\.(\d+)$", node)
            if m:
                self.layer_out[int(m.group(1))] = t.detach()[0]
            m = re.match(r"vision\.block\.(\d+)$", node)
            if m:
                self.vit_out[int(m.group(1))] = t.detach()
            if node == "embed":
                self.cache["embed"]["h"] = t.detach()[0]
            if self._keep_full(node):
                key = name.replace("model.", "", 1)
                self.tensors[key] = t.detach().to("cpu", torch.float16).contiguous()
        return hook

    def _rope_ids(self, mod, args, out):
        pos = args[1] if len(args) > 1 else None
        if pos is not None:
            p = pos.detach().cpu()
            if p.dim() == 3:  # [3, B, T]
                p = p[:, 0]
            self.previews["position_ids"] = p.numpy().astype(np.int32)

    # ------------------------------------------------------------------ vision attention
    def _vit_attn(self, i: int):
        def hook(mod, args, kwargs, out):
            self.busy = True
            try:
                return self._vit_attn_impl(i, mod, args, kwargs, out)
            finally:
                self.busy = False
        return hook

    def _vit_attn_impl(self, i, mod, args, kwargs, out):
        if True:
            h = args[0] if args else kwargs["hidden_states"]
            cos, sin = kwargs["position_embeddings"]
            cu = kwargs.get("cu_seqlens")
            mm = _modeling_module(mod)
            N = h.shape[0]
            with torch.no_grad():
                q, k, _ = mod.qkv(h).reshape(N, 3, mod.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
                q, k = mm.apply_rotary_pos_emb_vision(q, k, cos, sin)
                bounds = cu.tolist() if cu is not None else [0, N]
                received = torch.zeros(mod.num_heads, N, device=h.device)
                entropy = torch.zeros(mod.num_heads, device=h.device)
                mean_mat = None
                for a, b in zip(bounds[:-1], bounds[1:]):
                    qs, ks = q[a:b].transpose(0, 1).float(), k[a:b].transpose(0, 1).float()
                    p = torch.softmax(qs @ ks.transpose(1, 2) * mod.scaling, -1)  # [H, n, n]
                    received[:, a:b] = p.mean(1)
                    entropy += -(p * (p + 1e-12).log()).sum(-1).sum(-1)
                    if a == 0 and (b - a) <= 2304:
                        mean_mat = p.mean(0)
                entropy /= N
            self.previews[f"vit_attn_received.{i}"] = received.cpu().half().numpy()
            self.previews[f"vit_attn_entropy.{i}"] = entropy.cpu().float().numpy()
            if mean_mat is not None:
                self.previews[f"vit_attn_mean.{i}"] = mean_mat.cpu().half().numpy()

    # ------------------------------------------------------------------ text attention
    def _text_attn(self, i: int):
        def hook(mod, args, kwargs, out):
            self.busy = True
            try:
                return self._text_attn_impl(i, mod, args, kwargs, out)
            finally:
                self.busy = False
        return hook

    def _text_attn_impl(self, i, mod, args, kwargs, out):
        if True:
            h = kwargs.get("hidden_states", args[0] if args else None)
            cos, sin = kwargs["position_embeddings"]
            mm = _modeling_module(mod)
            B, T, _ = h.shape
            hd = mod.head_dim
            with torch.no_grad():
                qp = mod.q_proj(h)
                gated = qp.shape[-1] == 2 * mod.config.num_attention_heads * hd
                if gated:
                    q, gate = torch.chunk(qp.view(B, T, -1, hd * 2), 2, dim=-1)
                    g = torch.sigmoid(gate.float())  # [B,T,H,hd]
                    self.previews[f"attn_gate_mean.{i}"] = g[0].mean(-1).cpu().half().numpy()  # [T,H]
                else:
                    q = qp.view(B, T, -1, hd)
                q = mod.q_norm(q).transpose(1, 2)
                k = mod.k_norm(mod.k_proj(h).view(B, T, -1, hd)).transpose(1, 2)
                q, k = mm.apply_rotary_pos_emb(q, k, cos, sin)
                rep = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(rep, dim=1)
                H = q.shape[1]
                mask = torch.ones(T, T, dtype=torch.bool, device=h.device).tril()
                vis = self.info.get("vision_mask_t")
                vis = vis.to(h.device) if vis is not None else None
                ans_rows = self.info.get("answer_rows")
                last = self.info["last_prompt_row"]
                sink = torch.zeros(H, device=h.device)
                to_img = torch.zeros(H, int(vis.sum()) if vis is not None else 0, device=h.device)
                mean_mat = torch.zeros(T, T, device=h.device) if T <= self.policy.max_attn_tokens else None
                ent = torch.zeros(H, device=h.device)
                last_row = None
                chunk = max(1, int(2**26 // max(1, H * T)))
                n_ans = 0
                for s in range(0, T, chunk):
                    e = min(T, s + chunk)
                    sc = (q[0, :, s:e].float() @ k[0].float().transpose(1, 2)) * mod.scaling
                    sc = sc.masked_fill(~mask[s:e], float("-inf"))
                    p = torch.softmax(sc, -1)  # [H, c, T]
                    sink += p[:, :, 0].sum(1)
                    ent += -(p * (p + 1e-12).log()).nan_to_num().sum(-1).sum(-1)
                    if mean_mat is not None:
                        mean_mat[s:e] = p.mean(0)
                    if s <= last < e:
                        last_row = p[:, last - s]
                    if vis is not None and ans_rows is not None:
                        rows = [r - s for r in ans_rows if s <= r < e]
                        if rows:
                            to_img += p[:, rows][:, :, vis].sum(1)
                            n_ans += len(rows)
                sink /= T
                ent /= T
            pre = f"attn.{i}"
            self.previews[f"{pre}.sink"] = sink.cpu().float().numpy()
            self.previews[f"{pre}.entropy"] = ent.cpu().float().numpy()
            if last_row is not None:
                self.previews[f"{pre}.last_row"] = last_row.cpu().half().numpy()
            if n_ans:
                self.previews[f"{pre}.answer_to_image"] = (to_img / n_ans).cpu().half().numpy()
            if mean_mat is not None:
                self.previews[f"{pre}.mean"] = mean_mat.cpu().half().numpy()

    # ------------------------------------------------------------------ gated deltanet
    def _gdn(self, i: int):
        def hook(mod, args, kwargs, out):
            self.busy = True
            try:
                return self._gdn_impl(i, mod, args, kwargs, out)
            finally:
                self.busy = False
        return hook

    def _gdn_impl(self, i, mod, args, kwargs, out):
        if True:
            h = kwargs.get("hidden_states", args[0] if args else None)
            with torch.no_grad():
                B, T, _ = h.shape
                a = mod.in_proj_a(h).float()[0]
                b = mod.in_proj_b(h).float()[0]
                beta = torch.sigmoid(b)
                g = -mod.A_log.float().exp() * F.softplus(a + mod.dt_bias.float())
                alpha = g.exp()  # [T, Hv]
                self.previews[f"gdn.{i}.alpha"] = alpha.cpu().half().numpy()
                self.previews[f"gdn.{i}.beta"] = beta.cpu().half().numpy()
                if self.policy.gdn_state:
                    qkv = mod.in_proj_qkv(h).transpose(1, 2)  # [B, C, T]
                    w = mod.conv1d.weight  # [C,1,K]
                    qkv = F.conv1d(F.pad(qkv, (w.shape[-1] - 1, 0)), w, groups=qkv.shape[1])
                    qkv = F.silu(qkv).transpose(1, 2)[0].float()
                    q, k, v = torch.split(qkv, [mod.key_dim, mod.key_dim, mod.value_dim], -1)
                    k = k.view(T, mod.num_k_heads, mod.head_k_dim)
                    v = v.view(T, mod.num_v_heads, mod.head_v_dim)
                    k = F.normalize(k, dim=-1)
                    r = mod.num_v_heads // mod.num_k_heads
                    if r > 1:
                        k = k.repeat_interleave(r, dim=1)
                    S = torch.zeros(mod.num_v_heads, mod.head_k_dim, mod.head_v_dim, device=h.device)
                    snorm = torch.empty(T, mod.num_v_heads, device=h.device)
                    enorm = torch.empty(T, mod.num_v_heads, device=h.device)
                    for t in range(T):
                        S = S * alpha[t][:, None, None]
                        pred = torch.einsum("hkv,hk->hv", S, k[t])
                        err = (v[t] - pred) * beta[t][:, None]
                        S = S + k[t][:, :, None] * err[:, None, :]
                        snorm[t] = S.flatten(1).norm(dim=-1)
                        enorm[t] = err.norm(dim=-1)
                    self.previews[f"gdn.{i}.state_norm"] = snorm.cpu().half().numpy()
                    self.previews[f"gdn.{i}.write_norm"] = enorm.cpu().half().numpy()
                    if self.policy.full_tensors != "none" and i in self.sel_layers:
                        self.tensors[f"language_model.layers.{i}.linear_attn.final_state"] = S.cpu().half().contiguous()

    # ------------------------------------------------------------------ MoE router
    def _router(self, i: int):
        def hook(mod, args, out):
            logits, scores, idx = out
            with torch.no_grad():
                p = torch.softmax(logits.float(), -1)
                ent = -(p * (p + 1e-12).log()).sum(-1)
                load = torch.bincount(idx.reshape(-1), minlength=mod.num_experts)
            self.previews[f"moe.{i}.experts"] = idx.cpu().numpy().astype(np.int16)
            self.previews[f"moe.{i}.weights"] = scores.float().cpu().half().numpy()
            self.previews[f"moe.{i}.entropy"] = ent.cpu().half().numpy()
            self.previews[f"moe.{i}.load"] = load.cpu().numpy().astype(np.int32)
        return hook

    # ------------------------------------------------------------------ post-pass
    @torch.no_grad()
    def finalize(self) -> None:
        self.busy = True  # logit lens calls norm/lm_head: don't record those calls
        try:
            self._finalize()
        finally:
            self.busy = False

    def _finalize(self) -> None:
        if not self.layer_out:
            return
        L = max(self.layer_out) + 1
        stack = [self.layer_out[i].float() for i in range(L)]
        emb = self.cache["embed"].get("h")
        norms = torch.stack([s.norm(dim=-1) for s in stack])  # [L, T]
        self.previews["residual_norm"] = norms.cpu().half().numpy()
        prev = [emb.float()] + stack[:-1] if emb is not None else stack[:1] + stack[:-1]
        delta = torch.stack([(s - p).norm(dim=-1) / (p.norm(dim=-1) + 1e-6) for s, p in zip(stack, prev)])
        self.previews["residual_rel_update"] = delta.cpu().half().numpy()
        cos_next = torch.stack([F.cosine_similarity(s, p, dim=-1) for s, p in zip(stack, prev)])
        self.previews["residual_cos_prev"] = cos_next.cpu().half().numpy()
        if self.policy.pca:
            pcs = []
            for s in stack:
                x = s - s.mean(0, keepdim=True)
                try:
                    U, Sv, V = torch.pca_lowrank(x.cpu(), q=3, center=False)
                    pcs.append((x.cpu() @ V[:, :3]).numpy())
                except Exception:
                    pcs.append(np.zeros((x.shape[0], 3), np.float32))
            self.previews["residual_pca3"] = np.stack(pcs).astype(np.float16)
        if self.policy.logit_lens:
            lm = self.model.lm_head
            norm = self.model.model.language_model.norm
            rows = self.info["lens_rows"]
            ids, probs = [], []
            for s in stack:
                hrow = norm(s[rows].to(lm.weight.dtype))
                lp = torch.log_softmax(lm(hrow).float(), -1)
                v, ix = lp.topk(5, -1)
                ids.append(ix.cpu().numpy())
                probs.append(v.exp().cpu().numpy())
                tgt = self.info.get("lens_targets")
                if tgt is not None:
                    self.cache["lens_target_p"].setdefault("rows", []).append(
                        lp.gather(-1, tgt.to(lp.device)[:, None])[:, 0].exp().cpu().numpy())
            self.previews["logit_lens_ids"] = np.stack(ids).astype(np.int32)       # [L, R, 5]
            self.previews["logit_lens_probs"] = np.stack(probs).astype(np.float16)
            if "rows" in self.cache["lens_target_p"]:
                self.previews["logit_lens_target_p"] = np.stack(self.cache["lens_target_p"]["rows"]).astype(np.float16)  # [L, R]
        if self.vit_out:
            D = max(self.vit_out) + 1
            self.previews["vit_residual_norm"] = torch.stack(
                [self.vit_out[i].float().norm(dim=-1) for i in range(D)]).cpu().half().numpy()  # [D, N_p]
