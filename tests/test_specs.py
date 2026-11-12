import json
from pathlib import Path

import pytest

from lmmviz.specs.derive import build, vision_tokens

REF = json.loads((Path(__file__).parent / "hf_param_totals.json").read_text())
DATA = build()


@pytest.mark.parametrize("m", DATA["models"], ids=lambda m: m["key"])
def test_param_count_matches_hf_safetensors(m):
    ref = REF[m["key"]]
    ours = m["spec"]["params"]["total_with_mtp"]
    assert abs(ours - ref) / ref < 1e-6, (ours, ref)


def test_scope_is_at_most_100b():
    assert all(m["spec"]["params"]["total_with_mtp"] < 100e9 for m in DATA["models"])


def test_layer_patterns():
    by = {m["key"]: m["spec"] for m in DATA["models"]}
    lt = by["qwen3.5-0.8b"]["text"]["layer_types"]
    assert lt[:4] == ["linear_attention"] * 3 + ["full_attention"] and len(lt) == 24
    assert by["qwen3-vl-8b"]["vision"]["deepstack_visual_indexes"] == [8, 16, 24]
    assert by["qwen3.8-27b"]["vision"]["deepstack_visual_indexes"] == []


def test_vision_tokens_448():
    s = {m["key"]: m["spec"] for m in DATA["models"]}["qwen3-vl-8b"]
    v = vision_tokens(s, 448, 448)
    assert v["grid_thw"] == [1, 28, 28] and v["llm_tokens"] == 196
