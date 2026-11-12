"""Interventions: small, reversible edits applied to a loaded model (or its inputs) before a run.

Each intervention is a class with `apply(model)` returning itself and `remove()`.
Input-level interventions (image/processor edits) implement `transform_image` / `processor_kwargs`.
Specs come from experiment YAML, e.g. {"name": "head_ablate", "layer": 3, "heads": [0, 5]}.
"""
from __future__ import annotations

import re
import types

import torch
from PIL import Image, ImageFilter

REGISTRY: dict[str, type] = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def build(spec: dict | None):
    if not spec or spec.get("name") in (None, "none", "baseline"):
        return Baseline({})
    return REGISTRY[spec["name"]](spec)


def spec_id(spec: dict | None) -> str:
    if not spec or spec.get("name") in (None, "none", "baseline"):
        return "baseline"
    items = sorted(((str(k), v) for k, v in spec.items() if k != "name"))
    parts = [spec["name"]] + [f"{k}{'-'.join(map(str, v)) if isinstance(v, list) else v}" for k, v in items]
    return re.sub(r"[^A-Za-z0-9._\-]+", "", "_".join(parts))


class Intervention:
    name = "base"
    description = ""
    families: tuple[str, ...] = ("qwen3_vl", "qwen3_5", "qwen3_8")

    def __init__(self, spec: dict):
        self.spec = spec
        self.handles = []
        self._restore = []

    def apply(self, model):
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        for fn in self._restore:
            fn()
        self.handles, self._restore = [], []

    def transform_image(self, img: Image.Image) -> Image.Image:
        return img

    def processor_kwargs(self) -> dict:
        return {}

    def generation_kwargs(self) -> dict:
        return {}

    def text_layers(self, model):
        return model.model.language_model.layers

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description, **self.spec}


@register
class Baseline(Intervention):
    name = "baseline"
    description = "Unmodified model."


# ------------------------------------------------------------------ input level
@register
class Resolution(Intervention):
    name = "resolution"
    description = "Change the vision token budget via max_pixels (e.g. 100352 ≈ 128 tokens, 1605632 ≈ 2048 tokens)."

    def processor_kwargs(self):
        return {"max_pixels": int(self.spec["max_pixels"])}


@register
class BlankImage(Intervention):
    name = "blank_image"
    description = "Replace the image by a mid-grey canvas of the same size (does the answer need the image?)."

    def transform_image(self, img):
        return Image.new("RGB", img.size, (128, 128, 128))


@register
class BlurImage(Intervention):
    name = "blur_image"
    description = "Gaussian-blur the image (radius px): removes fine detail, keeps layout."

    def transform_image(self, img):
        return img.filter(ImageFilter.GaussianBlur(self.spec.get("radius", 8)))


@register
class Thinking(Intervention):
    name = "thinking"
    description = "Turn the chat-template thinking switch on/off (Qwen3.5 / Qwen3.8 hybrid models)."
    families = ("qwen3_5", "qwen3_8")

    def generation_kwargs(self):
        return {"enable_thinking": bool(self.spec.get("enabled", True)), "max_new_tokens": self.spec.get("max_new_tokens", 2048)}


# ------------------------------------------------------------------ vision side
@register
class DeepStackOff(Intervention):
    name = "deepstack_off"
    description = "Zero the DeepStack features so nothing from intermediate ViT layers is added to LLM layers 0-2."
    families = ("qwen3_vl",)

    def apply(self, model):
        for m in model.model.visual.deepstack_merger_list:
            self.handles.append(m.register_forward_hook(lambda mod, a, out: out * 0))
        return self


@register
class VisionTokenAblate(Intervention):
    name = "vision_ablate"
    description = "Replace merged vision embeddings by their mean (mode=mean) or shuffle their order (mode=shuffle)."

    def apply(self, model):
        mode = self.spec.get("mode", "mean")
        g = torch.Generator().manual_seed(0)

        def hook(mod, a, out):
            t = out.pooler_output if hasattr(out, "pooler_output") else out
            if mode == "mean":
                new = t.mean(0, keepdim=True).expand_as(t).clone()
            else:
                new = t[torch.randperm(t.shape[0], generator=g).to(t.device)]
            if hasattr(out, "pooler_output"):
                out.pooler_output = new
                return out
            return new
        self.handles.append(model.model.visual.register_forward_hook(hook))
        return self


@register
class MRoPEChunked(Intervention):
    name = "mrope_chunked"
    description = "Use Qwen2-VL-style contiguous t/h/w frequency chunks instead of interleaved MRoPE."

    def apply(self, model):
        rot = model.model.language_model.rotary_emb
        old = rot.recomposition_frequencies
        sec = rot.mrope_section

        def chunked(self_, freq):
            out = freq[0].clone()
            a = sec[0]
            out[..., a:a + sec[1]] = freq[1, ..., a:a + sec[1]]
            out[..., a + sec[1]:a + sec[1] + sec[2]] = freq[2, ..., a + sec[1]:a + sec[1] + sec[2]]
            return torch.cat((out, out), dim=-1)
        rot.recomposition_frequencies = types.MethodType(chunked, rot)
        self._restore.append(lambda: setattr(rot, "recomposition_frequencies", old))
        return self


@register
class TextOnlyPositions(Intervention):
    name = "flat_positions"
    description = "Give image tokens 1-D positions like text (t=h=w=running index): removes 2-D spatial positions."

    def apply(self, model):
        inner = model.model
        old = inner.get_rope_index

        def flat(*args, **kwargs):
            pos, deltas = old(*args, **kwargs)
            T = pos.shape[-1]
            ar = torch.arange(T, device=pos.device)
            return ar.view(1, 1, T).expand_as(pos).clone(), torch.zeros_like(deltas)
        inner.get_rope_index = flat
        self._restore.append(lambda: setattr(inner, "get_rope_index", old))
        return self


# ------------------------------------------------------------------ LLM side
@register
class HeadAblate(Intervention):
    name = "head_ablate"
    description = "Zero the output of selected attention heads in one full-attention layer (before o_proj)."

    def apply(self, model):
        layer = self.text_layers(model)[int(self.spec["layer"])]
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            raise ValueError("layer has no full attention (it is a Gated DeltaNet layer)")
        hd, heads = attn.head_dim, list(self.spec["heads"])

        def pre(mod, args):
            x = args[0].clone()
            for h in heads:
                x[..., h * hd:(h + 1) * hd] = 0
            return (x,)
        self.handles.append(attn.o_proj.register_forward_pre_hook(pre))
        return self


@register
class AttnGateOff(Intervention):
    name = "attn_gate_off"
    description = "Force the gated-attention sigmoid gate to 1 (plain attention) in all or some layers."
    families = ("qwen3_5", "qwen3_8")

    def apply(self, model):
        layers = self.spec.get("layers")
        for i, layer in enumerate(self.text_layers(model)):
            attn = getattr(layer, "self_attn", None)
            if attn is None or (layers and i not in layers):
                continue
            hd = attn.head_dim

            def hook(mod, a, out, hd=hd):
                o = out.clone()
                v = o.view(*o.shape[:-1], -1, 2 * hd)
                v[..., hd:] = 30.0  # sigmoid(30) == 1
                return o
            self.handles.append(attn.q_proj.register_forward_hook(hook))
        return self


@register
class SkipLayer(Intervention):
    name = "skip_layer"
    description = "Make decoder layer(s) the identity: output = input."

    def apply(self, model):
        for i in self.spec["layers"]:
            layer = self.text_layers(model)[int(i)]
            store = {}

            def pre(mod, args, kwargs, store=store):
                store["h"] = args[0] if args else kwargs["hidden_states"]

            def post(mod, args, kwargs, out, store=store):
                return store["h"] if isinstance(out, torch.Tensor) else (store["h"],) + tuple(out[1:])
            self.handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
            self.handles.append(layer.register_forward_hook(post, with_kwargs=True))
        return self


@register
class GDNNoForget(Intervention):
    name = "gdn_no_forget"
    description = "Gated DeltaNet: force decay α=1 (never forget) in the given linear-attention layers."
    families = ("qwen3_5", "qwen3_8")

    def apply(self, model):
        for i in self.spec["layers"]:
            la = self.text_layers(model)[int(i)].linear_attn
            self.handles.append(la.in_proj_a.register_forward_hook(lambda m, a, out: torch.full_like(out, -1e4)))
        return self


@register
class GDNNoWrite(Intervention):
    name = "gdn_no_write"
    description = "Gated DeltaNet: force write strength β=0 (state never updated) in the given layers."
    families = ("qwen3_5", "qwen3_8")

    def apply(self, model):
        for i in self.spec["layers"]:
            la = self.text_layers(model)[int(i)].linear_attn
            self.handles.append(la.in_proj_b.register_forward_hook(lambda m, a, out: torch.full_like(out, -1e4)))
        return self


@register
class MoETopK(Intervention):
    name = "moe_topk"
    description = "Change how many experts each token uses (e.g. 8 → 2)."

    def apply(self, model):
        for layer in self.text_layers(model):
            gate = getattr(layer.mlp, "gate", None)
            if gate is not None and hasattr(gate, "top_k"):
                old = gate.top_k
                gate.top_k = int(self.spec["k"])
                self._restore.append(lambda g=gate, o=old: setattr(g, "top_k", o))
        return self


@register
class MoENoShared(Intervention):
    name = "moe_no_shared"
    description = "Disable the always-on shared expert (Qwen3.5 MoE)."
    families = ("qwen3_5",)

    def apply(self, model):
        for layer in self.text_layers(model):
            se = getattr(layer.mlp, "shared_expert", None)
            if se is not None:
                self.handles.append(se.register_forward_hook(lambda m, a, out: out * 0))
        return self


def catalogue() -> list[dict]:
    return [{"name": c.name, "description": c.description, "families": list(c.families)} for c in REGISTRY.values()]
