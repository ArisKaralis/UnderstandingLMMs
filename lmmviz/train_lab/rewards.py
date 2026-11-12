"""Verifiable rewards for RL-VLIR-mini (no LLM judge inside the loop).

reward = correct (exact match, MLLM_Testing judge.extract rule)        ∈ {0, 1}
       + 0.5 · IoU(first predicted bbox_2d, gold box)                 ∈ [0, 0.5]
       + 0.1 · format (<think>…</think><answer>…</answer>)            ∈ {0, 0.1}
       − 0.1 · language mixing (CJK characters in an English prompt)   (Qwen3-VL uses a language-consistency penalty)
"""
from __future__ import annotations

import json
import re

from lmmviz.data import vlir

BOX = re.compile(r'\{\s*"bbox_2d"\s*:\s*\[([^\]]+)\]')
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower().rstrip("."))


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def first_box(text: str):
    m = BOX.search(text)
    if not m:
        return None
    try:
        v = [float(x) for x in m.group(1).split(",")]
        return v if len(v) == 4 else None
    except ValueError:
        return None


def reward(text: str, answers: list[str], gold_box: list[float] | None, prompt: str = "") -> dict:
    judge = vlir.mllm()["judge"]
    pred = judge.extract(text)
    correct = float(any(norm(pred) == norm(a) for a in answers if a))
    fmt = 0.1 if re.search(r"<think>.*</think>\s*<answer>.*</answer>", text, re.S) else 0.0
    b = first_box(text)
    box = 0.5 * iou(b, gold_box) if (b and gold_box) else 0.0
    lang = -0.1 if (CJK.search(text) and not CJK.search(prompt)) else 0.0
    parts = {"correct": correct, "iou": box, "format": fmt, "lang": lang}
    return {"total": sum(parts.values()), "parts": parts, "pred": pred, "box": b}


def to_1000(bbox01: list[float]) -> list[int]:
    return [round(1000 * v) for v in bbox01]
