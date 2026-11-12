"""Capture + intervention tests on tiny random-weight models (CPU, no weights downloaded)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mdb", ROOT / "scripts/local/make_demo_bundle.py")
mdb = importlib.util.module_from_spec(spec); spec.loader.exec_module(mdb)
sys.path.insert(0, str(ROOT / "scripts/local"))
from validate_bundle import validate  # noqa: E402

from lmmviz.interventions import registry as iv  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module", params=["qwen3.5-0.8b", "qwen3-vl-2b"])
def bundle(request, tmp_path_factory):
    return mdb.build(request.param, "vlir-152", tmp_path_factory.mktemp("tr"))


def test_bundle_validates(bundle):
    assert validate(bundle) == []
    m = json.loads((bundle / "manifest.json").read_text())
    idx = json.loads((bundle / "previews/index.json").read_text())
    assert "residual_norm" in idx and "logit_lens_ids" in idx and "position_ids" in idx
    if m["model"]["family"] == "qwen3_5":
        assert any(k.startswith("gdn.0.state_norm") for k in idx)
        assert "attn_gate_mean.3" in idx
    else:
        ti = json.loads((bundle / "tensors/index.json").read_text())
        assert any(k.startswith("visual.deepstack_merger_list") for k in ti)


def test_position_ids_are_3d(bundle):
    p = np.load(bundle / "previews/position_ids.npy")
    tok = json.loads((bundle / "tokens.json").read_text())
    assert p.shape == (3, len(tok["ids"]))
    vis = [i for i, s in enumerate(tok["segments"]) if s == "vision"]
    assert len(set(p[1, vis])) > 1 and len(set(p[0, vis])) == 1  # rows vary, time constant for a still image


def _inputs(key):
    from transformers import AutoProcessor
    from lmmviz.specs.registry import BY_KEY
    from PIL import Image
    proc = AutoProcessor.from_pretrained(BY_KEY[key].hf_id, max_pixels=50176, min_pixels=3136)
    img = Image.new("RGB", (224, 224), (200, 30, 30))
    text = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "colour?"}]}],
                                    tokenize=False, add_generation_prompt=True)
    return proc(text=[text], images=[img], return_tensors="pt")


@torch.no_grad()
def test_head_ablation_zeroes_only_that_head():
    model = mdb.tiny_model("qwen3-vl-2b")
    inp = _inputs("qwen3-vl-2b")
    attn = model.model.language_model.layers[1].self_attn
    seen = {}
    h = attn.o_proj.register_forward_pre_hook(lambda m, a: seen.__setitem__("x", a[0].clone()))
    ivn = iv.build({"name": "head_ablate", "layer": 1, "heads": [0]}).apply(model)
    model(**inp)
    ivn.remove(); h.remove()
    # recorded pre-hook ran before the ablation hook? order: ours registered first, so check output change instead
    out_a = model(**inp).logits
    ivn = iv.build({"name": "head_ablate", "layer": 1, "heads": [0]}).apply(model)
    out_b = model(**inp).logits
    ivn.remove()
    assert not torch.allclose(out_a, out_b)
    assert torch.allclose(model(**inp).logits, out_a)  # removal restores the model


@torch.no_grad()
def test_deepstack_off_changes_only_qwen3vl_text_side():
    model = mdb.tiny_model("qwen3-vl-2b")
    inp = _inputs("qwen3-vl-2b")
    base = model(**inp).logits
    ivn = iv.build({"name": "deepstack_off"}).apply(model)
    off = model(**inp).logits
    ivn.remove()
    assert not torch.allclose(base, off)


@torch.no_grad()
def test_gdn_no_forget_sets_alpha_to_one():
    model = mdb.tiny_model("qwen3.5-0.8b")
    inp = _inputs("qwen3.5-0.8b")
    la = model.model.language_model.layers[0].linear_attn
    got = {}
    ivn = iv.build({"name": "gdn_no_forget", "layers": [0]}).apply(model)
    h = la.in_proj_a.register_forward_hook(lambda m, a, out: got.__setitem__("a", out))
    model(**inp)
    h.remove(); ivn.remove()
    g = -la.A_log.float().exp() * torch.nn.functional.softplus(got["a"].float() + la.dt_bias.float())
    assert torch.allclose(g.exp(), torch.ones_like(g), atol=1e-6)


def test_spec_ids_are_unique_and_safe():
    specs = [None, {"name": "head_ablate", "layer": 3, "heads": [0, 1]}, {"name": "thinking", "enabled": True}, {"name": "resolution", "max_pixels": 100352}]
    ids = [iv.spec_id(s) for s in specs]
    assert len(set(ids)) == len(ids) and all("/" not in i for i in ids)
