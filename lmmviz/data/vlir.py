"""Samples and helpers shared with ../MLLM_Testing (reused, not copied).

Environment (set in scripts/hpc/cluster.env on the HPC):
    MLLM_TESTING_DIR  path to the MLLM_Testing checkout (default: ../MLLM_Testing)
    VLIR_ROOT         VLIR download root (default: $DATA_DIR/VLIR, else MLLM_Testing/data/VLIR)
"""
from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def mllm_dir() -> Path:
    return Path(os.environ.get("MLLM_TESTING_DIR", REPO.parent / "MLLM_Testing")).resolve()


def vlir_root() -> Path:
    if os.environ.get("VLIR_ROOT"):
        return Path(os.environ["VLIR_ROOT"])
    if os.environ.get("DATA_DIR") and (Path(os.environ["DATA_DIR"]) / "VLIR").exists():
        return Path(os.environ["DATA_DIR"]) / "VLIR"
    return mllm_dir() / "data" / "VLIR"


@functools.lru_cache
def mllm():
    """Import MLLM_Testing modules (r3eval.run, r3eval.judge, sft.make_data) from the sibling checkout."""
    root = mllm_dir()
    if not root.exists():
        raise FileNotFoundError(f"MLLM_Testing not found at {root}; set MLLM_TESTING_DIR")
    for p in (root / "scripts" / "r3eval", root / "scripts" / "sft", root / "scripts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import importlib
    run = importlib.import_module("run")          # r3eval/run.py: R3_PROMPT, STOPS, crop_region, smart_resize
    judge = importlib.import_module("judge")      # r3eval/judge.py: extract, rule
    make_data = importlib.import_module("make_data")  # sft/make_data.py: convert, is_dev
    return {"run": run, "judge": judge, "make_data": make_data}


def mllm_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(mllm_dir()), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


@dataclass
class Sample:
    id: str
    kind: str
    image_path: Path
    question: str
    answers: list[str]
    source: str
    note: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["image_path"] = str(self.image_path)
        return d


@functools.lru_cache
def _vlir_rows() -> dict[int, dict]:
    p = vlir_root() / "sft" / "vlir_sft_12k.json"
    return {r["sample_id"]: r for r in json.loads(p.read_text())}


def _bench_rows(bench: str) -> dict[str, dict]:
    base = mllm_dir()
    p = base / "data" / "r3eval" / f"{bench}.jsonl"
    if bench == "vlir_steer_test" and not p.exists():
        p = base / "data" / "vlir_steer" / "vlir_steer_test.jsonl"
    if not p.exists():  # fall back to the judged outputs, which carry the same input fields
        cands = sorted((base / "outputs" / "r3eval").glob(f"Qwen3-VL-8B-Instruct_*_{bench}.judged.jsonl"))
        if not cands:
            raise FileNotFoundError(f"no data for bench {bench}")
        p = cands[0]
    return {str(r["id"]): r for r in map(json.loads, p.open())}


def load_sample(spec: dict) -> Sample:
    if spec["kind"] == "vlir_sft":
        r = _vlir_rows()[int(spec["sample_id"])]
        name = Path(r["image_path"]).name
        img = vlir_root() / "sft" / "imgs" / name
        if not img.exists():  # curated examples carry their own copy
            alt = mllm_dir() / "outputs" / "examples" / "vlir" / f"{r['source']}_{r['sample_id']}" / "image.jpg"
            img = alt if alt.exists() else img
        return Sample(spec["id"], "vlir_sft", img, r["question"], list(r["answers"]), r["source"], spec.get("note", ""),
                      {"sample_id": r["sample_id"], "bbox_list": r.get("bbox_list"), "model_response": r.get("model_response")})
    if spec["kind"] == "bench":
        r = _bench_rows(spec["bench"])[str(spec["bench_id"])]
        img = (mllm_dir() / "data" / "r3eval" / r["image"]).resolve()
        answers = r.get("gold_answers") or [r.get("label")]
        return Sample(spec["id"], "bench", img, r["question"], [a for a in answers if a], r.get("source", spec["bench"]),
                      spec.get("note", ""), {"bench": spec["bench"], "label": r.get("label"), "pair": r.get("pair"),
                                             "ref_boxes": r.get("ref_boxes"), "group": r.get("group")})
    raise ValueError(spec["kind"])


def sample_set(name: str = "default") -> list[dict]:
    return yaml.safe_load((HERE / "sample_sets.yaml").read_text())[name]
