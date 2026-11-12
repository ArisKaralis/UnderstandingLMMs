"""Tiny random-weight versions of both architectures, captured with the real capture code on CPU.

Uses only the processor/tokenizer files of the real models (a few MB), never their weights, so it
runs anywhere and in seconds. Produces data/traces/demo__<family>__... bundles (manifest.demo = true).

    python scripts/local/make_demo_bundle.py [--out data/traces]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from lmmviz.capture import run as R
from lmmviz.capture.policy import get as get_policy
from lmmviz.data import vlir
from lmmviz.interventions import registry as iv
from lmmviz.specs.registry import BY_KEY

TINY = {
    "qwen3.5-0.8b": {
        "text": dict(num_hidden_layers=4, hidden_size=128, intermediate_size=256, num_attention_heads=4, num_key_value_heads=2,
                     head_dim=64, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
                     mtp_num_hidden_layers=0, layer_types=["linear_attention"] * 3 + ["full_attention"]),
        "vision": dict(depth=4, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=128),
    },
    "qwen3-vl-2b": {
        "text": dict(num_hidden_layers=4, hidden_size=128, intermediate_size=256, num_attention_heads=4, num_key_value_heads=2, head_dim=32),
        "vision": dict(depth=4, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=128, deepstack_visual_indexes=[1, 2, 3]),
    },
}


def tiny_model(key: str):
    cfg = AutoConfig.from_pretrained(BY_KEY[key].hf_id)
    for k, v in TINY[key]["text"].items():
        setattr(cfg.text_config, k, v)
    for k, v in TINY[key]["vision"].items():
        setattr(cfg.vision_config, k, v)
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(cfg, attn_implementation="sdpa").eval()
    return model


def build(key: str, sample_id: str, out: Path, max_pixels: int = 50176) -> Path:
    processor = AutoProcessor.from_pretrained(BY_KEY[key].hf_id, max_pixels=max_pixels, min_pixels=R.DEFAULT_MIN_PIXELS)
    model = tiny_model(key)
    spec = {s["id"]: s for s in vlir.sample_set("smoke")}[sample_id]
    sample = vlir.load_sample(spec)
    tkw = R.template_kwargs(processor, None)
    gen = R.run_direct(model, processor, sample, tkw, max_new_tokens=8)
    ivn = iv.build(None)
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="demo_", dir=out))  # never touch a real bundle of the same name
    path = R.trace(model, processor, gen, get_policy("full"), sample, key, "direct", ivn, tmp, tkw,
                   {"timing": {"load_s": 0.0, "generate_s": 0.0}, "demo": True})
    demo = out / ("demo__" + path.name)
    if demo.exists():
        shutil.rmtree(demo)
    path.rename(demo)
    shutil.rmtree(tmp, ignore_errors=True)
    m = json.loads((demo / "manifest.json").read_text())
    m["id"], m["demo"] = demo.name, True
    (demo / "manifest.json").write_text(json.dumps(m, indent=1))
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/traces"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for key in TINY:
        p = build(key, "vlir-152", a.out)
        print("wrote", p, f"{json.loads((p / 'manifest.json').read_text())['bytes'] / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
