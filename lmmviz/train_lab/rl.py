"""Tiny-scale RL on VLIR (GRPO / DAPO / GSPO / SAPO) with a full per-token record of selected steps.

Per optimiser step, for each of P prompts:
  1. sample G responses from the current policy (π_old)                  -> text, token ids
  2. reward each response (lmmviz.train_lab.rewards)                      -> r_i
  3. log-probs of every response token under π_old and π_ref              -> [G, L]
  4. `inner` gradient updates on the same rollouts; from the 2nd update on
     π_θ ≠ π_old, so ratios, clipping and gating become visible
  5. the last inner update is stored (and re-scored under all 4 algorithms) for the dashboard

    python -m lmmviz.train_lab.rl --model qwen3.5-0.8b --algo grpo --steps 4 --group 4 --prompts 1 --demo     # laptop
    python -m lmmviz.train_lab.rl --model qwen3-vl-2b --algo sapo --init-adapter data/training/<sft_run>/adapter  # HPC
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch
from PIL import Image

from lmmviz.data import vlir

from . import common as C
from .losses import AlgoConfig, dapo_keep_group, objective, recompute_all
from .rewards import reward, to_1000


def load_prompts(n: int, source: str, seed: int) -> list[dict]:
    """Prompts with gold answers and boxes. HPC: VLIR RL file (data_rgrpo.json, rows with a gold bbox);
    fallback / laptop: VLIR SFT rows (dev split excluded) whose box list gives the gold region."""
    rl_file = vlir.vlir_root() / "rl" / "data_rgrpo.json"
    out = []
    if source == "rl" and rl_file.exists():
        rows = json.loads(rl_file.read_text())
        for r in rows:
            img = r.get("image") or r.get("image_path")
            bbox = r.get("bbox")
            if not img or not bbox:
                continue
            p = vlir.vlir_root() / "rl" / "imgs" / img.split("/")[-1]
            if not p.exists():
                continue
            ans = r.get("answer") or r.get("answers") or r.get("solution")
            out.append({"id": str(r.get("id", len(out))), "image": p, "question": r.get("problem") or r.get("question"),
                        "answers": ans if isinstance(ans, list) else [str(ans)], "bbox_px": bbox})
    if not out:
        for r in C.vlir_sft_rows("train", None, seed=seed):
            out.append({"id": r["id"], "image": C.data_root() / r["image"], "question": r["question"], "answers": r["answers"],
                        "gold_box": r["boxes"][0] if r["boxes"] else None})
    random.Random(seed).shuffle(out)
    return out[:n]


def gold_box_1000(item: dict, img: Image.Image):
    if item.get("gold_box"):
        return item["gold_box"]
    if item.get("bbox_px"):
        x1, y1, x2, y2 = item["bbox_px"]
        return [round(1000 * x1 / img.width), round(1000 * y1 / img.height), round(1000 * x2 / img.width), round(1000 * y2 / img.height)]
    return None


def seq_logps(model, inputs: dict, resp_ids: torch.Tensor, grad: bool) -> torch.Tensor:
    """log π(y_t | x, y_<t) for every response token. inputs = processor output for the prompt."""
    ids = torch.cat([inputs["input_ids"], resp_ids[None]], 1)
    extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask") and not (v.dim() == 2 and v.shape[1] == inputs["input_ids"].shape[1])}
    per_tok = {k: torch.cat([v, torch.zeros(1, resp_ids.shape[0], dtype=v.dtype, device=v.device)], 1)
               for k, v in inputs.items() if k not in ("input_ids", "attention_mask") and v.dim() == 2 and v.shape[1] == inputs["input_ids"].shape[1]}
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), **extra, **per_tok)
        P = inputs["input_ids"].shape[1]
        logits = out.logits[0, P - 1:-1].float()
        return torch.log_softmax(logits, -1).gather(-1, resp_ids[:, None])[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--algo", default="grpo", choices=["grpo", "dapo", "gspo", "sapo"])
    ap.add_argument("--init-adapter", help="LoRA adapter from an SFT run to start from")
    ap.add_argument("--source", default="rl", choices=["rl", "sft"])
    ap.add_argument("--n-prompts", type=int, default=300)
    ap.add_argument("--prompts", type=int, default=2, help="prompts per optimiser step")
    ap.add_argument("--group", type=int, default=8, help="G responses per prompt")
    ap.add_argument("--inner", type=int, default=2, help="gradient updates per rollout batch")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-pixels", type=int, default=200704)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    device = C.pick_device()
    model, processor = C.load(a.model, device, a.max_pixels)
    from peft import PeftModel, get_peft_model
    if a.init_adapter:  # merge the SFT adapter so that π_ref (adapter disabled) == the SFT starting policy
        model = PeftModel.from_pretrained(model, a.init_adapter).merge_and_unload()
    model = get_peft_model(model, C.lora_config(r=16, alpha=32))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)
    cfg = AlgoConfig.preset(a.algo)
    R = vlir.mllm()["run"]
    tok = processor.tokenizer
    tkw = {"enable_thinking": False} if "enable_thinking" in str(processor.chat_template or "") else {}
    prompts = load_prompts(a.n_prompts, a.source, a.seed)
    if not prompts:
        raise SystemExit("no RL prompts found")
    rid = a.run_id or f"rl__{a.model}__{a.algo}__{time.strftime('%m%d-%H%M')}"
    w = C.RunWriter(rid, {"kind": "rl", "model": a.model, "algo": a.algo, "algo_config": cfg.to_dict(), "config": vars(a),
                          "n_prompts": len(prompts), "demo": a.demo,
                          "reward": "correct + 0.5·IoU(box) + 0.1·format − 0.1·language-mix"})
    want = C.record_steps(a.steps)
    cursor = 0
    for step in range(a.steps):
        t0 = time.time()
        groups, stats = [], {"reward": [], "correct": [], "iou": [], "format": [], "len": []}
        for _ in range(a.prompts):
            item = prompts[cursor % len(prompts)]; cursor += 1
            img = Image.open(item["image"]).convert("RGB")
            text = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": item["question"] + R.R3_PROMPT}]}],
                                                 tokenize=False, add_generation_prompt=True, **tkw)
            inputs = processor(text=[text], images=[img], return_tensors="pt", add_special_tokens=False).to(model.device)
            model.eval()
            with torch.no_grad():
                gen = model.generate(**inputs, do_sample=True, temperature=a.temperature, top_p=1.0, max_new_tokens=a.max_new_tokens,
                                     num_return_sequences=a.group, stop_strings=["</answer>"], tokenizer=tok)
            P = inputs["input_ids"].shape[1]
            gold = gold_box_1000(item, img)
            resps = []
            for g in gen[:, P:]:
                ids = g[g != tok.pad_token_id] if tok.pad_token_id is not None else g
                txt = tok.decode(ids, skip_special_tokens=True)
                rw = reward(txt, item["answers"], gold, item["question"])
                resps.append({"ids": ids, "text": txt, "reward": rw["total"], "reward_parts": rw["parts"], "pred": rw["pred"], "box": rw["box"]})
                for k in ("correct", "iou", "format"):
                    stats[k].append(rw["parts"][k])
                stats["reward"].append(rw["total"]); stats["len"].append(len(ids))
            rewards = torch.tensor([r["reward"] for r in resps])
            keep = a.algo != "dapo" or dapo_keep_group(rewards)
            with torch.no_grad():
                old = [seq_logps(model, inputs, r["ids"], False) for r in resps]
                with model.disable_adapter():
                    ref = [seq_logps(model, inputs, r["ids"], False) for r in resps]
            groups.append({"item": item, "img": img, "inputs": inputs, "resps": resps, "rewards": rewards, "old": old, "ref": ref, "keep": keep, "gold": gold})
        model.train()
        last = None
        for it in range(a.inner):
            opt.zero_grad(set_to_none=True)
            losses = []
            for gi, grp in enumerate(groups):
                if not grp["keep"]:
                    continue
                L = max(len(r["ids"]) for r in grp["resps"])
                G = len(grp["resps"])
                logp = torch.zeros(G, L, device=model.device)
                lold = torch.zeros(G, L, device=model.device); lref = torch.zeros(G, L, device=model.device)
                mask = torch.zeros(G, L, device=model.device)
                rows = []
                for i, r in enumerate(grp["resps"]):
                    n = len(r["ids"])
                    lp = seq_logps(model, grp["inputs"], r["ids"], True)
                    rows.append(lp)
                    lold[i, :n] = grp["old"][i]; lref[i, :n] = grp["ref"][i]; mask[i, :n] = 1
                logp = torch.stack([torch.nn.functional.pad(x, (0, L - x.shape[0])) for x in rows])
                o = objective(logp, lold, mask, grp["rewards"].to(model.device), cfg, lref)
                (o["loss"] / max(1, len(groups))).backward()
                losses.append(float(o["loss"].detach()))
                if it == a.inner - 1 and gi == 0:
                    last = (grp, logp.detach(), lold, lref, mask, o)
            gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0)) if losses else 0.0
            if losses:
                opt.step()
        if step in want and last is not None:
            grp, lp, lo, lr_, mk, o = last
            item = grp["item"]
            n = [len(r["ids"]) for r in grp["resps"]]
            w.step(step, {
                "question": item["question"], "gold": item["answers"], "gold_bbox": grp["gold"],
                "image": w.image(f"{item['id']}", grp["img"]), "algo": a.algo,
                "group": [{"text": r["text"], "tokens": [tok.decode([t]) for t in r["ids"].tolist()], "reward": r["reward"],
                           "reward_parts": r["reward_parts"], "pred": r["pred"], "box": r["box"],
                           "logp": lp[i, :n[i]].tolist(), "logp_old": lo[i, :n[i]].tolist(), "logp_ref": lr_[i, :n[i]].tolist()}
                          for i, r in enumerate(grp["resps"])],
                "recomputed": recompute_all(lp, lo, mk, grp["rewards"].to(lp.device), lr_),
            })
        mean = lambda x: float(sum(x) / max(1, len(x)))
        kl = float(last[5]["kl"].sum() / last[4].sum()) if last and "kl" in last[5] else None
        w.metric({"step": step, "reward": mean(stats["reward"]), "accuracy": mean(stats["correct"]), "box_iou_reward": mean(stats["iou"]),
                  "format": mean(stats["format"]), "response_len": mean(stats["len"]), "loss": mean(losses) if losses else 0.0,
                  "clip_fraction": float(last[5]["clip_fraction"]) if last else 0.0, "kl": kl if kl is not None else 0.0,
                  "grad_norm": gn, "groups_kept": sum(g["keep"] for g in groups), "seconds": time.time() - t0})
    w.finish({"steps": a.steps})
    model.save_pretrained(w.dir / "adapter")
    print(f"done: {w.dir}")


if __name__ == "__main__":
    main()
