"""Tiny on-policy distillation (strong-to-weak, as in Qwen3 / Qwen3-VL §4.3).

The student samples a response; the teacher (same tokenizer family) scores every student token;
the student minimises the per-token reverse KL(student ‖ teacher) over the full vocabulary.

    python -m lmmviz.train_lab.distill --student qwen3-vl-2b --teacher qwen3-vl-8b --steps 50     # HPC (1×80GB)
Step records: per-token KL, and the top-5 student vs teacher distributions at each position.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from PIL import Image

from lmmviz.data import vlir

from . import common as C
from .rl import load_prompts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", default="qwen3-vl-2b")
    ap.add_argument("--teacher", default="qwen3-vl-8b")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-pixels", type=int, default=200704)
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--run-id")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    device = C.pick_device()
    student, processor = C.load(a.student, device, a.max_pixels)
    teacher, _ = C.load(a.teacher, device, a.max_pixels)
    teacher.eval().requires_grad_(False)
    from peft import get_peft_model
    student = get_peft_model(student, C.lora_config(r=16, alpha=32))
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)
    tok = processor.tokenizer
    R = vlir.mllm()["run"]
    prompts = load_prompts(a.n_prompts, "sft", 0)
    rid = a.run_id or f"distill__{a.student}__from__{a.teacher}__{time.strftime('%m%d-%H%M')}"
    w = C.RunWriter(rid, {"kind": "distill", "model": a.student, "teacher": a.teacher, "algo": "on-policy KL", "config": vars(a), "demo": a.demo})
    want = C.record_steps(a.steps)
    for step in range(a.steps):
        t0 = time.time()
        item = prompts[step % len(prompts)]
        img = Image.open(item["image"]).convert("RGB")
        text = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": item["question"] + R.R3_PROMPT}]}],
                                             tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt", add_special_tokens=False).to(student.device)
        P = inputs["input_ids"].shape[1]
        student.eval()
        with torch.no_grad():
            seq = student.generate(**inputs, do_sample=True, temperature=1.0, max_new_tokens=a.max_new_tokens)
        resp = seq[0, P:]
        full = {**inputs, "input_ids": seq, "attention_mask": torch.ones_like(seq)}
        per_tok = {k: torch.cat([v, torch.zeros(1, resp.shape[0], dtype=v.dtype, device=v.device)], 1)
                   for k, v in inputs.items() if k not in ("input_ids", "attention_mask") and v.dim() == 2 and v.shape[1] == P}
        full.update(per_tok)
        with torch.no_grad():
            t_logits = teacher(**full).logits[0, P - 1:-1].float()
        student.train()
        s_logits = student(**full).logits[0, P - 1:-1].float()
        s_lp, t_lp = F.log_softmax(s_logits, -1), F.log_softmax(t_logits, -1)
        kl = (s_lp.exp() * (s_lp - t_lp)).sum(-1)          # reverse KL per token  [L]
        loss = kl.mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step in want:
            st, si = s_lp.exp().topk(5, -1); tt, ti = t_lp.exp().topk(5, -1)
            w.step(step, {"question": item["question"], "image": w.image(item["id"], img), "tokens": [tok.decode([t]) for t in resp.tolist()],
                          "kl": kl.tolist(), "student_top": [[[tok.decode([int(i)]), float(p)] for i, p in zip(a_, b_)] for a_, b_ in zip(si.tolist(), st.tolist())],
                          "teacher_top": [[[tok.decode([int(i)]), float(p)] for i, p in zip(a_, b_)] for a_, b_ in zip(ti.tolist(), tt.tolist())]})
        w.metric({"step": step, "kl": float(loss), "response_len": int(resp.shape[0]), "seconds": time.time() - t0})
    w.finish({"steps": a.steps})


if __name__ == "__main__":
    main()
