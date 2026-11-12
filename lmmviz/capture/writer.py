"""Write / read trace bundles (format documented in docs/bundle_format.md)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file

BUNDLE_VERSION = 1


def write_bundle(out: Path, *, manifest: dict, tokens: dict, graph: dict, stats: list[dict],
                 tensors: dict[str, torch.Tensor], previews: dict[str, np.ndarray], outputs: dict,
                 images: list) -> Path:
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "previews").mkdir(parents=True)
    (tmp / "tensors").mkdir()
    (tmp / "images").mkdir()
    for i, im in enumerate(images):
        im.convert("RGB").save(tmp / "images" / f"{i}.png")
    (tmp / "tokens.json").write_text(json.dumps(tokens))
    (tmp / "graph.json").write_text(json.dumps(graph))
    (tmp / "outputs.json").write_text(json.dumps(outputs))
    if stats:
        cols = sorted({k for r in stats for k in r})
        table = pa.table({c: [r.get(c) for r in stats] for c in cols})
        pq.write_table(table, tmp / "stats.parquet")
    # group full tensors per top-level block so the backend can lazy-load one file at a time
    groups: dict[str, dict[str, torch.Tensor]] = {}
    for k, t in tensors.items():
        parts = k.split(".")
        g = ".".join(parts[:3]) if parts[0] in ("language_model", "visual") and len(parts) > 2 and parts[2].isdigit() else parts[0]
        groups.setdefault(g, {})[k] = t
    index = {}
    for g, d in groups.items():
        fn = f"{g}.safetensors"
        save_file(d, str(tmp / "tensors" / fn))
        for k, t in d.items():
            index[k] = {"file": fn, "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}
    (tmp / "tensors" / "index.json").write_text(json.dumps(index))
    pindex = {}
    for k, a in previews.items():
        np.save(tmp / "previews" / f"{k}.npy", a)
        pindex[k] = {"shape": list(a.shape), "dtype": str(a.dtype)}
    (tmp / "previews" / "index.json").write_text(json.dumps(pindex))
    size = sum(f.stat().st_size for f in tmp.rglob("*") if f.is_file())
    manifest = {**manifest, "bundle_version": BUNDLE_VERSION, "bytes": size, "complete": True}
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return out


def is_complete(out: Path) -> bool:
    m = out / "manifest.json"
    return m.exists() and json.loads(m.read_text()).get("complete", False)
