"""Run one (model, sample, mode, intervention) and write a trace bundle.

    # one run
    python -m lmmviz.capture.run --model qwen3.5-0.8b --sample vlir-152 --mode direct --policy standard
    # experiment grid (used by scripts/hpc/capture_array.slurm)
    python -m lmmviz.capture.run --experiment configs/experiments/baseline.yaml --list
    python -m lmmviz.capture.run --experiment configs/experiments/baseline.yaml --index 3

Steps: load model → render prompt → generate (direct, or the VLM-R³ crop loop reused from
MLLM_Testing) → one teacher-forced forward pass over prompt + answer with all recorders
attached → write bundle (manifest, tokens, graph, stats, tensors, previews, outputs, images).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from lmmviz.data import vlir
from lmmviz.graph.build_graph import build_graph
from lmmviz.interventions import registry as iv
from lmmviz.specs.derive import normalise
from lmmviz.specs.registry import BY_KEY

from .policy import get as get_policy
from .recorders import Recorder
from .writer import is_complete, write_bundle

DEFAULT_MAX_PIXELS = 401408  # MLLM_Testing default: ~512 vision tokens after 2x2 merge
DEFAULT_MIN_PIXELS = 32 * 28 * 28


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(hf_id: str, device: str, max_pixels: int, min_pixels: int = DEFAULT_MIN_PIXELS):
    if device == "mps":  # threaded weight loading onto MPS intermittently segfaults (transformers 5.17)
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(hf_id, max_pixels=max_pixels, min_pixels=min_pixels)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    kw = {"device_map": "auto"} if device == "cuda" and torch.cuda.device_count() > 1 else {"device_map": device}
    model = AutoModelForImageTextToText.from_pretrained(hf_id, dtype=dtype, attn_implementation="sdpa", **kw).eval()
    return model, processor


def template_kwargs(processor, thinking: bool | None) -> dict:
    has = "enable_thinking" in str(processor.chat_template or "")
    return {"enable_thinking": bool(thinking)} if has and thinking is not None else ({"enable_thinking": False} if has else {})


def render_prefix(processor, question: str, tkw: dict) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **tkw)


@torch.inference_mode()
def gen_step(model, processor, text: str, images: list, max_new_tokens: int, stops: list[str] | None):
    tok = processor.tokenizer
    inputs = processor(text=[text], images=images, return_tensors="pt", add_special_tokens=False).to(model.device)
    n_prompt = int(inputs["input_ids"].shape[1])
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, output_scores=True,
                         return_dict_in_generate=True, **({"stop_strings": stops, "tokenizer": tok} if stops else {}))
    new = out.sequences[0, n_prompt:].tolist()
    steps = []
    for s, tid in zip(out.scores, new):
        lp = torch.log_softmax(s[0].float(), -1)
        v, ix = lp.topk(10)
        steps.append({"id": tid, "token": tok.decode([tid]), "logprob": float(lp[tid]),
                      "top": [[int(i), tok.decode([int(i)]), float(x)] for i, x in zip(ix.tolist(), v.tolist())]})
    decoded = tok.decode(new, skip_special_tokens=True)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    return decoded, bool(new) and new[-1] in eos, n_prompt, steps


def run_direct(model, processor, sample, tkw, max_new_tokens):
    img = Image.open(sample.image_path).convert("RGB")
    prefix = render_prefix(processor, sample.question, tkw)
    text, _, n_prompt, steps = gen_step(model, processor, prefix, [img], max_new_tokens, None)
    return {"prefix": prefix, "reply_parts": [text], "images": [img], "steps": [steps], "rounds": 1,
            "boxes": [], "prompt_tokens": [n_prompt]}


def run_r3(model, processor, sample, tkw, max_new_tokens, max_rounds=8, crop_max_pixels=DEFAULT_MAX_PIXELS):
    """Port of MLLM_Testing r3eval Runner.r3 (same prompt, stop strings, crop rule); keeps per-step logprobs."""
    R = vlir.mllm()["run"]
    ip = processor.image_processor
    factor = ip.patch_size * ip.merge_size
    min_px = getattr(ip, "min_pixels", None) or DEFAULT_MIN_PIXELS
    max_px = getattr(ip, "max_pixels", None) or DEFAULT_MAX_PIXELS
    raw = Image.open(sample.image_path).convert("RGB")
    h, w = R.smart_resize(raw.height, raw.width, factor, min_px, max_px)
    fed = raw.resize((w, h))
    frame = (1000, 1000)  # Qwen3+ use relative 0-1000 boxes
    prefix = render_prefix(processor, sample.question + R.R3_PROMPT, tkw)
    parts, images, steps_all, boxes, prompt_tokens = [], [fed], [], [], []
    last_crop, rounds = None, 0
    while rounds < max_rounds:
        rounds += 1
        text = prefix + ("".join(parts) + "\n" if parts else "")
        gen, eos, n_prompt, steps = gen_step(model, processor, text, images, max_new_tokens, R.STOPS)
        cuts = [gen.find(s) for s in R.STOPS if s in gen]
        if cuts:
            gen = gen[: min(cuts)]
        steps_all.append(steps)
        prompt_tokens.append(n_prompt)
        if "<answer>" in gen and "</answer>" not in gen:
            gen += "</answer>\n"
        if eos or "</answer>" in gen:
            parts.append(gen)
            break
        if "bbox_2d" in gen:
            gen += "]}\n"
        add_image, op = False, None
        m = R.JSON_OBJECT.findall(gen)
        if m:
            try:
                op = json.loads(m[-1])
            except json.JSONDecodeError:
                op = None
        if op:
            try:
                region = R.crop_region(raw, op["bbox_2d"], frame, factor, min_px, crop_max_pixels)
                boxes.append(op["bbox_2d"])
                if region is not None:
                    last_crop = region
                add_image = last_crop is not None
            except Exception:
                gen += "Invalid operation!"
        parts.append(gen)
        if add_image:
            parts.append(R.VISION)
            images.append(last_crop)
    return {"prefix": prefix, "reply_parts": parts, "images": images, "steps": steps_all, "rounds": rounds,
            "boxes": boxes, "prompt_tokens": prompt_tokens}


# ------------------------------------------------------------------ token bookkeeping
def segment_tokens(processor, ids: list[int], n_prompt: int, cfg) -> list[str]:
    tok = processor.tokenizer
    special = set(tok.all_special_ids)
    img_id = cfg.image_token_id
    vs, ve = cfg.vision_start_token_id, cfg.vision_end_token_id
    think_open = tok.convert_tokens_to_ids("<think>")
    think_close = tok.convert_tokens_to_ids("</think>")
    segs, role, in_think = [], "system", False
    i = 0
    while i < len(ids):
        t = ids[i]
        if t == img_id:
            segs.append("vision")
        elif t in (vs, ve):
            segs.append("vision_marker")
        elif t == think_open:
            in_think = True; segs.append("special")
        elif t == think_close:
            in_think = False; segs.append("special")
        elif tok.convert_ids_to_tokens(t) == "<|im_start|>":
            segs.append("special")
            if i + 1 < len(ids):
                role = tok.decode([ids[i + 1]]).strip()
        elif t in special:
            segs.append("special")
        elif i >= n_prompt:
            segs.append("thinking" if in_think else "answer")
        else:
            segs.append({"user": "user", "system": "system", "assistant": "assistant_prefill"}.get(role, "user"))
        i += 1
    return segs


def patch_raster_order(grid_thw: list[int], merge: int) -> list[int]:
    """Index k in the ViT sequence -> raster index (row*W + col) of that 16px patch."""
    t, h, w = grid_thw
    out = []
    for ti in range(t):
        for bh in range(h // merge):
            for bw in range(w // merge):
                for i in range(merge):
                    for j in range(merge):
                        out.append(ti * h * w + (bh * merge + i) * w + (bw * merge + j))
    return out


def trace(model, processor, gen: dict, policy, sample, model_key: str, mode: str, ivn, out_root: Path,
          tkw: dict, extra_manifest: dict) -> Path:
    cfg = model.config
    full_text = gen["prefix"] + "".join(gen["reply_parts"])
    inputs = processor(text=[full_text], images=gen["images"], return_tensors="pt", add_special_tokens=False).to(model.device)
    n_prompt = int(processor(text=[gen["prefix"]], images=gen["images"][:1], return_tensors="pt",
                             add_special_tokens=False)["input_ids"].shape[1])
    ids = inputs["input_ids"][0].tolist()
    T = len(ids)
    segs = segment_tokens(processor, ids, n_prompt, cfg)
    vis_mask = torch.tensor([s == "vision" for s in segs])
    ans_rows = [i - 1 for i, s in enumerate(segs) if s in ("answer", "thinking") and i > 0]  # rows that PREDICT answer tokens
    non_vis = [i for i, s in enumerate(segs) if s != "vision"]
    lens_rows = non_vis[-512:]
    info = {"vision_mask_t": vis_mask if vis_mask.any() else None, "answer_rows": ans_rows or None,
            "last_prompt_row": n_prompt - 1, "lens_rows": torch.tensor(lens_rows),
            "lens_targets": torch.tensor([ids[r + 1] if r + 1 < T else ids[r] for r in lens_rows])}
    rec = Recorder(model, policy, info).attach()
    t0 = time.time()
    with torch.inference_mode():
        out = model(**inputs, use_cache=False)
    rec.finalize()
    rec.detach()
    secs = time.time() - t0
    # answer-token logprobs under teacher forcing
    lp = torch.log_softmax(out.logits[0].float(), -1)
    tgt = torch.tensor(ids[1:], device=lp.device)
    tok_lp = lp[:-1].gather(-1, tgt[:, None])[:, 0].cpu().numpy()
    rec.previews["token_logprob"] = np.concatenate([[0.0], tok_lp]).astype(np.float32)
    ent = -(lp.exp() * lp).sum(-1).cpu().numpy()
    rec.previews["token_entropy"] = ent.astype(np.float16)
    del out, lp

    # ---- token + image bookkeeping
    tok = processor.tokenizer
    grids = inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else []
    vc = cfg.vision_config
    merge, patch = vc.spatial_merge_size, vc.patch_size
    images_meta, cursor = [], 0
    starts = [i for i, s in enumerate(segs) if s == "vision" and (i == 0 or segs[i - 1] != "vision")]
    for k, (g, im) in enumerate(zip(grids, gen["images"])):
        n_tok = g[0] * g[1] * g[2] // merge ** 2
        images_meta.append({"index": k, "grid_thw": g, "resized": [g[1] * patch, g[2] * patch], "original": [im.height, im.width],
                            "n_patches": g[0] * g[1] * g[2], "n_tokens": n_tok, "token_start": starts[k] if k < len(starts) else None,
                            "patch_offset": cursor, "raster_order": patch_raster_order(g, merge)})
        cursor += g[0] * g[1] * g[2]
    tokens = {"ids": ids, "strings": [tok.decode([i]) for i in ids], "segments": segs, "n_prompt": n_prompt,
              "images": images_meta, "lens_rows": lens_rows,
              "special": {"image": cfg.image_token_id, "vision_start": cfg.vision_start_token_id, "vision_end": cfg.vision_end_token_id}}
    rec.previews["lens_target_ids"] = info["lens_targets"].numpy().astype(np.int32)

    # ---- graph with this run's concrete shapes (first image; T matches the real sequence)
    spec = normalise(json.loads(cfg.to_json_string()), BY_KEY[model_key].family)
    n_vis_total = int(vis_mask.sum())
    n_v0 = images_meta[0]["n_tokens"] if images_meta else 0
    hw0 = tuple(images_meta[0]["resized"]) if images_meta else None
    graph = build_graph(spec, n_text=T - n_v0 - 2 if images_meta else T, img_hw=hw0)

    reply = "".join(p for p in gen["reply_parts"] if p != vlir.mllm()["run"].VISION) if mode == "r3" else gen["reply_parts"][0]
    judge = vlir.mllm()["judge"]
    try:
        extracted = judge.extract(reply)
        score, score_src = judge.rule({"response": reply, "label": "/".join(sample.answers)})
    except Exception:
        extracted, score, score_src = None, None, "error"
    lens_vocab = {}
    if "logit_lens_ids" in rec.previews:
        uniq = np.unique(rec.previews["logit_lens_ids"])
        lens_vocab = {int(i): tok.decode([int(i)]) for i in uniq}
    outputs = {"lens_vocab": lens_vocab, "reply": reply, "parts": gen["reply_parts"], "rounds": gen["rounds"], "boxes": gen["boxes"],
               "steps": gen["steps"], "prompt_tokens": gen["prompt_tokens"], "answer_extracted": extracted,
               "rule_score": score, "rule_source": score_src, "gold": sample.answers}
    m = BY_KEY[model_key]
    bundle_id = run_id(model_key, sample.id, mode, ivn.spec)
    manifest = {
        "id": bundle_id, "model": m.to_dict(), "sample": sample.to_dict(), "mode": mode, "intervention": ivn.to_dict(),
        "intervention_id": iv.spec_id(ivn.spec), "policy": policy.to_dict(), "template_kwargs": tkw,
        "prompt_text": gen["prefix"], "full_text": full_text, "T": T, "n_vision_tokens": n_vis_total,
        "timing": {"trace_forward_s": secs, **extra_manifest.get("timing", {})},
        "env": {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                "device": str(model.device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
                "python": platform.python_version()},
        "git": {"lmmviz": _git(Path(__file__).resolve().parents[2]), "mllm_testing": vlir.mllm_commit()},
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **{k: v for k, v in extra_manifest.items() if k != "timing"},
    }
    return write_bundle(out_root / bundle_id, manifest=manifest, tokens=tokens, graph=graph, stats=rec.stats,
                        tensors=rec.tensors, previews=rec.previews, outputs=outputs, images=gen["images"])


def _git(path: Path) -> str | None:
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def run_id(model_key: str, sample_id: str, mode: str, spec: dict | None) -> str:
    return f"{model_key}__{sample_id}__{mode}__{iv.spec_id(spec)}"


# ------------------------------------------------------------------ experiments
def expand(exp: dict) -> list[dict]:
    samples = exp.get("samples") or [s["id"] for s in vlir.sample_set(exp.get("sample_set", "default"))]
    ivs = exp.get("interventions") or [None]
    rows = []
    for model, sample, mode, ivspec in itertools.product(exp["models"], samples, exp.get("modes", ["direct"]), ivs):
        fam = BY_KEY[model].family
        if ivspec and ivspec.get("name") in iv.REGISTRY and fam not in iv.REGISTRY[ivspec["name"]].families:
            continue
        rows.append({"model": model, "sample": sample, "mode": mode, "intervention": ivspec,
                     "policy": exp.get("policy", "standard"), "max_new_tokens": exp.get("max_new_tokens", 512),
                     "max_pixels": exp.get("max_pixels", DEFAULT_MAX_PIXELS), "thinking": exp.get("thinking")})
    return rows


def execute(rows: list[dict], out_root: Path, force: bool = False) -> None:
    spec_by_id = {s["id"]: s for name in ("default", "smoke") for s in vlir.sample_set(name)}
    device = pick_device()
    cur = None
    for r in rows:
        bid = run_id(r["model"], r["sample"], r["mode"], r["intervention"])
        if not force and is_complete(out_root / bid):
            print(f"skip (complete) {bid}")
            continue
        ivn = iv.build(r["intervention"])
        max_px = ivn.processor_kwargs().get("max_pixels", r["max_pixels"])
        key = (r["model"], max_px)
        if cur is None or cur[0] != key:
            cur = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            t0 = time.time()
            model, processor = load_model(BY_KEY[r["model"]].hf_id, device, max_px)
            cur = (key, model, processor, time.time() - t0)
        _, model, processor, load_s = cur
        sample = vlir.load_sample(spec_by_id[r["sample"]])
        gk = ivn.generation_kwargs()
        tkw = template_kwargs(processor, gk.get("enable_thinking", r.get("thinking")))
        max_new = gk.get("max_new_tokens", r["max_new_tokens"])
        ivn.apply(model)
        orig_open = Image.open
        try:
            if type(ivn).transform_image is not iv.Intervention.transform_image:
                def patched(*a, **k):
                    return ivn.transform_image(orig_open(*a, **k).convert("RGB"))
                Image.open = patched
            t0 = time.time()
            gen = (run_r3 if r["mode"] == "r3" else run_direct)(model, processor, sample, tkw, max_new)
            gen_s = time.time() - t0
        finally:
            Image.open = orig_open
        try:
            path = trace(model, processor, gen, get_policy(r["policy"]), sample, r["model"], r["mode"], ivn, out_root, tkw,
                         {"timing": {"load_s": load_s, "generate_s": gen_s}, "run_spec": r})
        finally:
            ivn.remove()
        print(f"wrote {path}  ({json.loads((path / 'manifest.json').read_text())['bytes']/1e6:.1f} MB)  reply={gen['reply_parts'][-1][:80]!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--sample")
    ap.add_argument("--mode", default="direct", choices=["direct", "r3"])
    ap.add_argument("--policy", default="standard", choices=["lite", "standard", "full"])
    ap.add_argument("--intervention", default=None, help='JSON, e.g. \'{"name":"head_ablate","layer":3,"heads":[0]}\'')
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--experiment", type=Path)
    ap.add_argument("--index", type=int, help="row of the expanded experiment (Slurm array task id)")
    ap.add_argument("--shard", help="i/n: run every n-th row starting at i")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("TRACE_DIR", "data/traces")))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.experiment:
        rows = expand(yaml.safe_load(a.experiment.read_text()))
        if a.list:
            for i, r in enumerate(rows):
                print(i, run_id(r["model"], r["sample"], r["mode"], r["intervention"]))
            print(f"{len(rows)} runs")
            return
        if a.index is not None:
            rows = [rows[a.index]]
        elif a.shard:
            i, n = map(int, a.shard.split("/"))
            rows = rows[i::n]
    else:
        rows = [{"model": a.model, "sample": a.sample, "mode": a.mode, "policy": a.policy,
                 "intervention": json.loads(a.intervention) if a.intervention else None,
                 "max_new_tokens": a.max_new_tokens, "max_pixels": a.max_pixels, "thinking": None}]
    a.out.mkdir(parents=True, exist_ok=True)
    execute(rows, a.out, a.force)


if __name__ == "__main__":
    main()
