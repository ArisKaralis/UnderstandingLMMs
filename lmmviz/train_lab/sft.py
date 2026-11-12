"""Tiny-scale LoRA SFT on VLIR with a full per-token record of selected steps.

Same recipe as MLLM_Testing/scripts/sft/train_lora.py (VLIRDataset with crops re-cut by the r3
crop rule, loss on everything after the prompt except image tokens, LoRA r=64/α=128 on the LM's
q/k/v/o/gate/up/down, frozen ViT) — but a plain loop so every step can be inspected.

    # laptop smoke run
    python -m lmmviz.train_lab.sft --model qwen3.5-0.8b --data vlir --n-train 8 --steps 6 --demo
    # HPC (scripts/hpc/train_sft.slurm)
    python -m lmmviz.train_lab.sft --model qwen3-vl-2b --data vlir --n-train 500 --epochs 1
    python -m lmmviz.train_lab.sft --model qwen3-vl-2b --data steer_edited --epochs 5   # CoT-steering replay
"""
from __future__ import annotations

import argparse
import math
import time

import torch

from lmmviz.capture.run import segment_tokens

from . import common as C


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--data", default="vlir", choices=["vlir", "steer_edited", "steer_original"])
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=None, help="cap on optimiser steps")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-pixels", type=int, default=401408)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--full-ft", action="store_true", help="full fine-tuning of the LLM instead of LoRA")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    device = C.pick_device()
    model, processor = C.load(a.model, device, a.max_pixels)
    tl = C.train_lora_module()

    rows = C.vlir_sft_rows("train", a.n_train, seed=a.seed) if a.data == "vlir" else C.steer_rows(a.data.split("_")[1])
    path = C.write_rows(C.OUT_ROOT / "_tmp" / f"{a.data}_{a.model}.jsonl", rows)
    ds = tl.VLIRDataset(path, C.data_root(), processor, a.max_pixels, a.max_length)

    for p in model.model.visual.parameters():
        p.requires_grad_(False)
    if a.full_ft:
        for n, p in model.named_parameters():
            p.requires_grad_("language_model" in n)
        method = "full"
    else:
        from peft import get_peft_model
        model = get_peft_model(model, C.lora_config())
        method = "lora r=64 α=128"
    model.train()
    if device != "mps":
        model.gradient_checkpointing_enable() if hasattr(model, "gradient_checkpointing_enable") else None
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = a.steps or max(1, math.ceil(len(ds) * a.epochs / a.grad_accum))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.03 * total))) * 0.5 * (1 + math.cos(math.pi * s / total)))
    rid = a.run_id or f"sft__{a.model}__{a.data}__{time.strftime('%m%d-%H%M')}"
    w = C.RunWriter(rid, {"kind": "sft", "model": a.model, "algo": "sft", "data": a.data, "n_train": len(ds), "method": method,
                          "config": vars(a), "demo": a.demo, "trainable_params": sum(p.numel() for p in params)})
    probe = tl.collate([ds[0]])
    want = C.record_steps(total)
    tok = processor.tokenizer
    i = 0
    for step in range(total):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        loss_sum, ntok = 0.0, 0
        for _ in range(a.grad_accum):
            batch = tl.collate([ds[i % len(ds)]]); i += 1
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            (out.loss / a.grad_accum).backward()
            loss_sum += float(out.loss.detach()); ntok += int((batch["labels"] != -100).sum())
        gn = C.grad_norms_by_group(model)
        total_gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        rec = None
        if step in want:
            rec = {"loss": loss_sum / a.grad_accum, "lr": sched.get_last_lr()[0], "grad_norms": gn}
        opt.step(); sched.step()
        if rec is not None:
            rec["lora_delta"] = C.lora_delta_norms(model) if not a.full_ft else None
            rec["examples"] = [probe_record(model, probe, tok, processor)]
            w.step(step, rec)
        w.metric({"step": step, "loss": loss_sum / a.grad_accum, "lr": sched.get_last_lr()[0], "grad_norm": total_gn,
                  "label_tokens": ntok, "seconds": time.time() - t0})
    w.finish({"steps": total})
    out_dir = w.dir / "adapter"
    if not a.full_ft:
        model.save_pretrained(out_dir)
    print(f"done: {w.dir}")


@torch.no_grad()
def probe_record(model, batch, tok, processor) -> dict:
    """Per-token loss on a fixed probe example (same example at every recorded step)."""
    was = model.training
    model.eval()
    b = {k: v.to(model.device) for k, v in batch.items()}
    logits = model(**{k: v for k, v in b.items() if k != "labels"}).logits[0].float()
    model.train(was)
    ids = b["input_ids"][0]
    labels = b["labels"][0]
    lp = torch.log_softmax(logits[:-1], -1)
    tl = -lp.gather(-1, ids[1:, None])[:, 0]
    mask = (labels[1:] != -100)
    token_loss = [None] + [float(x) if m else None for x, m in zip(tl.tolist(), mask.tolist())]
    ids_l = ids.tolist()
    n_prompt = int((labels == -100).cumprod(0).sum())
    cfg = model.config if hasattr(model.config, "image_token_id") else model.base_model.model.config
    return {"tokens": [tok.decode([t]) for t in ids_l], "label_mask": [0] + [int(m) for m in mask.tolist()],
            "token_loss": token_loss, "segments": segment_tokens(processor, ids_l, n_prompt, cfg)}


if __name__ == "__main__":
    main()
