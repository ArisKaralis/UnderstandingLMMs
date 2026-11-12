"""Check trace bundles before/after copying them from the HPC.

    python scripts/local/validate_bundle.py data/traces/*        # exit code 1 on any error
Checks: files present, manifest complete, captured shapes agree with graph.json
(layer outputs [1,T,d], vision tokens, patch counts), previews/tensors indexes resolve.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

REQUIRED = ["manifest.json", "tokens.json", "graph.json", "outputs.json", "stats.parquet",
            "previews/index.json", "tensors/index.json"]


def validate(b: Path) -> list[str]:
    errs = [f"missing {f}" for f in REQUIRED if not (b / f).exists()]
    if errs:
        return errs
    m = json.loads((b / "manifest.json").read_text())
    if not m.get("complete"):
        errs.append("manifest not complete")
    g = json.loads((b / "graph.json").read_text())
    nodes = {n["id"]: n for n in g["nodes"]}
    tok = json.loads((b / "tokens.json").read_text())
    T = len(tok["ids"])
    if T != m["T"]:
        errs.append(f"tokens {T} != manifest T {m['T']}")
    if g["context"]["T"] != T:
        errs.append(f"graph T {g['context']['T']} != {T}")
    st = pq.read_table(b / "stats.parquet").to_pandas()
    d = nodes["embed"]["outputs"]["h₀"][1][1]
    for _, r in st[st.node.str.match(r"^llm\.layer\.\d+$")].iterrows():
        if list(r["shape"]) != [1, T, d]:
            errs.append(f"{r['node']} shape {list(r['shape'])} != [1,{T},{d}]")
    n_vis = sum(s == "vision" for s in tok["segments"])
    mg = st[st.node == "vision.merger"]
    if len(mg):
        total = sum(int(s[0]) for s in mg["shape"])
        if total != n_vis:
            errs.append(f"merger tokens {total} != vision tokens in sequence {n_vis}")
    pe = st[st.node == "vision.patch_embed"]
    if len(pe):
        n_p = sum(i["n_patches"] for i in tok["images"])
        if int(pe.iloc[0]["shape"][0]) != n_p:
            errs.append(f"patch_embed rows {pe.iloc[0]['shape'][0]} != patches {n_p}")
    if tok["images"] and "vision" in nodes:
        gv = g["context"]["vision"]
        if gv["grid_thw"] != tok["images"][0]["grid_thw"]:
            errs.append(f"graph grid {gv['grid_thw']} != captured {tok['images'][0]['grid_thw']}")
    for k in json.loads((b / "previews" / "index.json").read_text()):
        if not (b / "previews" / f"{k}.npy").exists():
            errs.append(f"preview file missing: {k}")
    for k, v in json.loads((b / "tensors" / "index.json").read_text()).items():
        if not (b / "tensors" / v["file"]).exists():
            errs.append(f"tensor file missing: {v['file']}")
    n_nonfinite = int(st["n_nonfinite"].sum())
    if n_nonfinite:
        errs.append(f"{n_nonfinite} non-finite values in recorded tensors")
    return errs


def main(paths: list[str]) -> int:
    bad = 0
    for p in map(Path, paths):
        if not (p / "manifest.json").exists():
            continue
        e = validate(p)
        print(("OK   " if not e else "FAIL ") + p.name)
        for x in e:
            print("     -", x)
        bad += bool(e)
    print(f"{bad} bundle(s) with errors")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
