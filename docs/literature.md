# Literature notes behind the dashboard

Scope: open models ≤100B. Numbers are from configs and papers. "Not disclosed" means that no primary source gives it.

## Qwen3-VL (arXiv 2511.21631, Nov 2025)
- **Architecture:** SigLIP-2 ViT (SO-400M for 8B and up: depth 27, d 1152; Large for 2B/4B: depth 24, d 1024), patch 16, temporal patch 2, 2D-RoPE plus interpolated absolute position embeddings. The 2×2 merger (2-layer MLP) maps patches to LLM width. Qwen3 dense or MoE LLM.
- **DeepStack:** features from ViT blocks [8,16,24] ([5,11,17] for 2B/4B), each through its own merger, are added to the hidden states of LLM layers 0–2 at the image positions. Ablation: average 74.7 → 76.0.
- **Interleaved MRoPE:** mrope_section [24,20,20] interleaved over 64 rotary pairs, θ = 5e6. **Text timestamps** (`<3.0 seconds>`) replace Qwen2.5-VL's absolute-time MRoPE.
- **Pre-training:**

  | Stage | Trains | Tokens | Sequence length |
  |---|---|---|---|
  | S0 | merger only | 67B | 8K |
  | S1 | all | 1T | 8K |
  | S2 | all | 1T | 32K |
  | S3 | all | 100B | 256K |

  Loss is a square-root-normalised per-token loss.
- **Post-training:**
  - SFT on about 1.2M samples (⅓ text): one epoch at 32K, then one at 256K.
  - Long-CoT cold start.
  - Strong-to-weak distillation, off-policy then on-policy KL.
  - Reasoning RL with **SAPO**: about 30K queries, 16 rollouts, pass rate >90% dropped.
  - General RL with rule and judge rewards.
  - Thinking-with-images agentic RL with answer, multi-turn and tool-call rewards.

## Qwen3.5 (Feb–Mar 2026) and Qwen3.8 (Aug 2026)
- **Layers:** hybrid 3:1 Gated DeltaNet / gated full attention (`layer_types`, full attention every 4th layer).
- **Full attention:** head_dim 256, partial rotary 0.25, mrope [11,11,10] interleaved, θ = 1e7. A sigmoid output gate is packed into q_proj (`attn_output_gate`).
- **Gated DeltaNet:** 16 QK heads × 128, 16–64 V heads × 128, short conv k = 4, α = exp(−e^A·softplus(a + dt_bias)), β = σ(b), gated RMSNorm output.
- **Other:** DeepStack removed. Vocab 248,320. MTP with 1 layer. Context 262K (about 1M with YaRN ×4).
- **MoE (35B-A3B):** 256 experts, top-8 plus a sigmoid-gated shared expert.
- **Qwen3.8-27B:** same class `Qwen3_5ForConditionalGeneration`, built on Qwen3.6-27B, adds `reasoning_effort` and `preserve_thinking`.
- **Post-training methods:** not disclosed. The model cards only say "RL across million-agent environments" and "asynchronous RL".

## Qwen3 (arXiv 2505.09388)
- Four stages: long-CoT cold start → reasoning RL (GRPO, 3,995 queries) → thinking-mode fusion (`<think></think>`, empty think block for non-thinking replies) → general RL (20+ tasks).
- Thinking budget.
- Strong-to-weak distillation: on-policy distillation of Qwen3-8B costs 1,800 GPU-h vs 17,920 GPU-h for RL.

## RL objectives (implemented in `lmmviz/train_lab/losses.py` and `frontend/src/lib/rl.ts`)
- **GRPO** (2402.03300): group-standardised advantage; token ratio with clip ε = 0.2; k3 KL estimator, β = 0.04; per-sequence mean, then group mean.
- **DAPO** (2503.14476): clip-higher 0.2 / 0.28; dynamic sampling; token-level normalisation; soft overlong penalty (16384 / 4096). AIME: naive GRPO 30 → DAPO 50.
- **GSPO** (2507.18071): length-normalised sequence ratio s_i = (π/π_old)^{1/|y|}; sequence clip 3e-4 / 4e-4; stable for MoE without Routing Replay.
- **SAPO** (2511.20347): f(r) = (4/τ)σ(τ(r−1)), gradient weight sech²(τ(r−1)/2); τ_pos = 1.0, τ_neg = 1.05. Used for Qwen3-VL.
- **DPO** (used for Qwen2.5-VL): −log σ(β[Δ log-ratio of chosen − rejected]).

## Building blocks
- Gated DeltaNet (2412.06464): S_t = (α_t I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ; chunkwise WY/UT training.
- Gated Attention (2505.06708): Y ⊙ σ(XW) after SDPA, head-specific; first-token attention 46.7% → 4.8%.
- MTP (DeepSeek-V3, 2412.19437): h' = M[RMSNorm(h); RMSNorm(Emb(t_{i+1}))] → block → shared head; drafts for speculative decoding.
- VLM-R³ (2505.16192): the region crop loop and the VLIR dataset used for the samples and the training lab.

## Out of scope (>100B)
Qwen3-VL-235B-A22B, Qwen3.5-122B-A10B and 397B-A17B, Qwen3.8-Flash-Next (125B-A6B; its tech report introduces QSA sparse attention, the Gated Residual 4-branch stream, n-gram embeddings and Muon), Qwen3.8-2.4T-A95B.
