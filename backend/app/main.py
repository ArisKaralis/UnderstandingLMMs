"""Dashboard API: specs, architecture graphs (with playground overrides), trace bundles, diffs, training runs.

    uvicorn backend.app.main:app --reload --port 8000
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from lmmviz.graph.build_graph import apply_overrides, build_graph
from lmmviz.interventions.registry import catalogue
from lmmviz.specs.derive import build as build_specs, per_token_costs, text_params, vision_params

from . import store

app = FastAPI(title="Inside Qwen-VL")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _nf(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except (FileNotFoundError, KeyError) as e:
            raise HTTPException(404, f"not found: {e}")
        except ValueError as e:
            raise HTTPException(400, str(e))
    return wrapper


@functools.lru_cache
def specs() -> dict:
    return build_specs(store.ROOT / "data" / "specs" / "configs")


def spec_of(model: str) -> dict:
    for m in specs()["models"]:
        if m["key"] == model:
            return m["spec"]
    raise KeyError(model)


@app.get("/api/specs")
def api_specs():
    return specs()


@app.get("/api/graph/{model}")
@_nf
def api_graph(model: str, n_text: int = 48, h: int = 448, w: int = 448, no_image: bool = False,
              overrides: str | None = Query(None, description='JSON, e.g. {"text.num_layers": 12}')):
    s = spec_of(model)
    if overrides:
        s = apply_overrides(s, json.loads(overrides))
    g = build_graph(s, n_text=n_text, img_hw=None if no_image else (h, w))
    tp = text_params(s["text"])
    vp = vision_params(s["vision"]) if s.get("vision") else {"total": 0}
    g["summary"] = {"params_total": tp["total"] + vp["total"], "params_active": tp["active"] + vp["total"],
                    "text": tp, "vision": vp, "costs": per_token_costs(s), "spec": s}
    return g


@app.get("/api/interventions")
def api_interventions():
    return catalogue()


@app.get("/api/traces")
def api_traces(model: str | None = None, sample: str | None = None):
    t = store.list_traces()
    return [x for x in t if (not model or x["model"] == model) and (not sample or x["sample"] == sample)]


@app.get("/api/traces/{tid}/{part}")
@_nf
def api_trace_part(tid: str, part: str):
    files = {"manifest": "manifest.json", "tokens": "tokens.json", "graph": "graph.json", "outputs": "outputs.json",
             "previews": "previews/index.json", "tensors": "tensors/index.json"}
    if part == "stats":
        return store.stats(tid)
    if part not in files:
        raise KeyError(part)
    return store.read_json(tid, files[part])


@app.get("/api/traces/{tid}/preview/{key}")
@_nf
def api_preview(tid: str, key: str, row: int | None = None, rows: str | None = None, stride: int = 1):
    return store.encode(store.preview(tid, key), row, rows, stride)


@app.get("/api/traces/{tid}/tensor/{name}")
@_nf
def api_tensor(tid: str, name: str, row: int | None = None, rows: str | None = None, cols: int = 64):
    a = store.tensor(tid, name)
    x = a[0] if a.ndim == 3 and a.shape[0] == 1 else a
    if row is not None or rows:
        out = store.encode(x, row, rows, max_elems=2_000_000)
        return out
    corner = x[:16, :cols] if x.ndim >= 2 else x[:cols]
    return {"shape": list(a.shape), "dtype": str(a.dtype), "corner": np.asarray(corner, np.float32).tolist(),
            "row_norms": np.linalg.norm(x.reshape(-1, x.shape[-1]).astype(np.float32), axis=-1).tolist() if x.ndim >= 2 else None}


@app.get("/api/traces/{tid}/image/{k}")
@_nf
def api_image(tid: str, k: int):
    p = store.bundle(tid) / "images" / f"{int(k)}.png"
    if not p.exists():
        raise FileNotFoundError(p.name)
    return FileResponse(p)


@app.get("/api/diff")
@_nf
def api_diff(a: str, b: str):
    """Baseline-vs-variant comparison for the Experiments page."""
    out = {}
    for name, tid in (("a", a), ("b", b)):
        o = store.read_json(tid, "outputs.json")
        m = store.read_json(tid, "manifest.json")
        rn = store.preview(tid, "residual_norm")
        out[name] = {"id": tid, "reply": o["reply"], "score": o.get("rule_score"), "T": m["T"],
                     "intervention": m["intervention"], "residual_norm_mean": np.asarray(rn, np.float32).mean(1).tolist(),
                     "n_vision_tokens": m["n_vision_tokens"],
                     "first_step_top": (o["steps"][0][0]["top"] if o["steps"] and o["steps"][0] else None)}
    ta, tb = store.read_json(a, "tokens.json"), store.read_json(b, "tokens.json")
    if ta["ids"][:ta["n_prompt"]] == tb["ids"][:tb["n_prompt"]]:
        ra, rb = np.asarray(store.preview(a, "residual_norm"), np.float32), np.asarray(store.preview(b, "residual_norm"), np.float32)
        n = min(ra.shape[1], rb.shape[1], ta["n_prompt"])
        out["residual_norm_rel_change"] = (np.abs(rb[:, :n] - ra[:, :n]) / (ra[:, :n] + 1e-6)).mean(1).tolist()
    return out


@app.get("/api/training")
def api_training():
    return store.list_training()


@app.get("/api/training/{rid}/{name:path}")
@_nf
def api_training_file(rid: str, name: str):
    d = store.training_dir(rid)
    p = store._safe(d, name)
    if p.suffix == ".json":
        return json.loads(p.read_text())
    if p.suffix == ".jsonl":
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if p.suffix == ".npy":
        return store.encode(np.load(p))
    if p.is_dir():
        return sorted(x.name for x in p.iterdir())
    return FileResponse(p)


DIST = store.ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "unknown API route")
        f = (DIST / path).resolve()
        inside = DIST.resolve() in f.parents
        return FileResponse(f if path and inside and f.is_file() else DIST / "index.html")
