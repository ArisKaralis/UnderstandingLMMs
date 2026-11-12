"""Read-only access to trace bundles and training runs on disk (lazy, cached)."""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = Path(os.environ.get("TRACE_DIR", ROOT / "data" / "traces"))
TRAIN_DIR = Path(os.environ.get("TRAIN_DIR", ROOT / "data" / "training"))


def _safe(base: Path, name: str) -> Path:
    p = (base / name).resolve()
    if base.resolve() not in p.parents and p != base.resolve():
        raise FileNotFoundError(name)
    return p


def list_traces() -> list[dict]:
    out = []
    if not TRACE_DIR.exists():
        return out
    for d in sorted(TRACE_DIR.iterdir()):
        m = d / "manifest.json"
        if d.is_dir() and m.exists():
            j = json.loads(m.read_text())
            if j.get("complete"):
                out.append({"id": j["id"], "model": j["model"]["key"], "family": j["model"]["family"],
                            "sample": j["sample"]["id"], "mode": j["mode"], "intervention": j["intervention_id"],
                            "T": j["T"], "n_vision_tokens": j["n_vision_tokens"], "bytes": j["bytes"],
                            "policy": j["policy"]["name"], "demo": j.get("demo", False)})
    out.sort(key=lambda r: (r["demo"], r["model"], r["sample"], r["intervention"] != "baseline", r["id"]))
    return out


def bundle(tid: str) -> Path:
    p = _safe(TRACE_DIR, tid)
    if not (p / "manifest.json").exists():
        raise FileNotFoundError(tid)
    return p


@functools.lru_cache(maxsize=64)
def read_json(tid: str, name: str) -> dict:
    return json.loads((bundle(tid) / name).read_text())


@functools.lru_cache(maxsize=16)
def stats(tid: str) -> list[dict]:
    t = pq.read_table(bundle(tid) / "stats.parquet").to_pandas()
    recs = t.to_dict(orient="records")
    for r in recs:
        for k, v in list(r.items()):
            if isinstance(v, np.ndarray):
                r[k] = v.tolist()
            elif isinstance(v, float) and not np.isfinite(v):
                r[k] = None
    return recs


def preview(tid: str, key: str) -> np.ndarray:
    return np.load(_safe(bundle(tid) / "previews", f"{key}.npy"), mmap_mode="r")


def tensor(tid: str, name: str) -> np.ndarray:
    from safetensors import safe_open
    idx = read_json(tid, "tensors/index.json")
    if name not in idx:
        raise KeyError(name)
    with safe_open(str(bundle(tid) / "tensors" / idx[name]["file"]), framework="np") as f:
        return f.get_tensor(name)


def encode(a: np.ndarray, row: int | None = None, rows: str | None = None, stride: int = 1,
           max_elems: int = 4_000_000) -> dict:
    """Array -> JSON-friendly dict. `row` selects a[row]; `rows` = 'a:b'; `stride` subsamples the last two axes."""
    x = np.asarray(a)
    if row is not None:
        x = x[row]
    if rows:
        s, e = (int(v) if v else None for v in rows.split(":"))
        x = x[s:e]
    if stride > 1 and x.ndim >= 2:
        x = x[..., ::stride, ::stride]
    if x.size > max_elems:
        raise ValueError(f"array of {x.size} elements; use row/rows/stride")
    xf = x.astype(np.float32) if x.dtype.kind == "f" else x
    data = np.nan_to_num(xf).ravel().tolist() if x.dtype.kind == "f" else xf.ravel().tolist()
    return {"shape": list(x.shape), "dtype": str(a.dtype), "data": data}


def list_training() -> list[dict]:
    out = []
    if TRAIN_DIR.exists():
        for d in sorted(TRAIN_DIR.iterdir()):
            m = d / "run.json"
            if m.exists():
                out.append(json.loads(m.read_text()) | {"id": d.name})
    return out


def training_dir(rid: str) -> Path:
    p = _safe(TRAIN_DIR, rid)
    if not (p / "run.json").exists():
        raise FileNotFoundError(rid)
    return p
