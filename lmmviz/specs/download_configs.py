"""Download the small JSON files (config, preprocessor, generation config) for every model.

No weights are downloaded. Run on the laptop or the HPC login node:
    python -m lmmviz.specs.download_configs [--out data/specs/configs]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from .registry import MODELS

FILES = ["config.json", "preprocessor_config.json", "generation_config.json", "video_preprocessor_config.json"]


def download(out: Path, keys: list[str] | None = None) -> None:
    for m in MODELS:
        if keys and m.key not in keys:
            continue
        d = out / m.key
        d.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            try:
                p = hf_hub_download(m.hf_id, name)
            except EntryNotFoundError:
                continue
            except Exception as e:  # network / gated repo
                print(f"  ! {m.hf_id}/{name}: {type(e).__name__}: {e}")
                continue
            shutil.copy(p, d / name)
        ok = (d / "config.json").exists()
        print(f"{'ok ' if ok else 'ERR'} {m.key:24s} {m.hf_id}")
        if ok:
            json.loads((d / "config.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/specs/configs")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    download(Path(a.out), a.only)


if __name__ == "__main__":
    main()
