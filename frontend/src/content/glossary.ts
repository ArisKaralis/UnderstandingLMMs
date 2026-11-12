export const GLOSSARY: { term: string; kind?: string; text: string }[] = [
  { term: "Patch", kind: "preprocess", text: "A 16×16-pixel square of the image (×2 frames for still images, so 2×16×16×3 = 1536 numbers). Each patch becomes one vector inside the vision encoder." },
  { term: "Dynamic resolution", text: "The image keeps (roughly) its own size and aspect ratio: it is resized to multiples of 32 px within min/max pixel limits, so the number of patches — and of LLM tokens — depends on the image." },
  { term: "Patch merger", kind: "merger", text: "Concatenates each 2×2 block of neighbouring patch vectors and maps them with a 2-layer MLP to the LLM width. 4 patches → 1 visual token, so a 448×448 image gives 784 patches but only 196 LLM tokens." },
  { term: "ViT block", kind: "vit_block", text: "Vision Transformer layer: LayerNorm → bidirectional multi-head self-attention (every patch sees every patch of the same image) → LayerNorm → GELU MLP, each with a residual connection." },
  { term: "DeepStack", text: "Qwen3-VL only: features from three intermediate ViT layers are merged separately and ADDED to the hidden states of the first three LLM layers at the image positions. Adds fine detail without adding tokens. Removed in Qwen3.5." },
  { term: "Early fusion", kind: "embedding", text: "Image tokens are written into the same sequence as text tokens (placeholder rows of the embedding are overwritten), and the model is trained on interleaved multimodal data from the start (Qwen3.5)." },
  { term: "MRoPE", kind: "rope", text: "Multimodal rotary position embedding: every token gets a 3-D position (time, height, width). Text uses the same value on all three axes; image tokens use their row/column. 'Interleaved' (Qwen3-VL+) spreads t/h/w across all frequency bands instead of contiguous chunks." },
  { term: "Partial rotary", text: "Qwen3.5/3.8 rotate only the first 64 of the 256 head dimensions (factor 0.25); the remaining dimensions carry position-free content." },
  { term: "GQA", kind: "attention", text: "Grouped-query attention: many query heads share a few key/value heads (e.g. 32 Q / 8 KV). Cuts the KV cache by the group factor with little quality loss." },
  { term: "QK-norm", kind: "norm", text: "RMSNorm applied to each query and key head vector before RoPE; keeps attention logits bounded and training stable (Qwen3 family)." },
  { term: "Gated attention", kind: "gate", text: "Qwen3.5/3.8 full-attention layers multiply each head's output by a sigmoid gate computed from the token's hidden state. Adds non-linearity, gives sparse head outputs and removes the 'attention sink' on the first token." },
  { term: "Attention sink", text: "Many attention heads dump most of their probability on the first token when they have nothing useful to attend to. Measured here as mean attention to position 0." },
  { term: "Gated DeltaNet", kind: "recurrence", text: "A linear-attention layer with a fixed-size memory matrix per head (128×128). Each token decays the memory (α), measures what the memory predicts for its key, and writes only the error scaled by β (delta rule). No KV cache, cost linear in length. 3 of every 4 layers in Qwen3.5/3.8." },
  { term: "Short convolution", kind: "conv", text: "Depthwise causal 1-D convolution over the last 4 tokens applied to q, k, v in Gated DeltaNet layers: a cheap local n-gram-like feature before the recurrence." },
  { term: "RMSNorm", kind: "norm", text: "Divide a vector by its root-mean-square and scale per channel. Qwen3.5 uses zero-centred weights (scale = 1 + w)." },
  { term: "SwiGLU MLP", kind: "mlp", text: "down(SiLU(gate(x)) ⊙ up(x)): a gated feed-forward layer that expands to the intermediate size and back." },
  { term: "Mixture of Experts", kind: "experts", text: "The FFN is replaced by many small experts; a router picks top-k per token (8 of 128 in Qwen3-VL-30B-A3B, 8 of 256 + 1 shared in Qwen3.5-35B-A3B). Total parameters are large, active parameters per token are small." },
  { term: "Router", kind: "router", text: "Linear layer scoring each expert for a token; softmax → top-k → renormalised weights." },
  { term: "Residual stream", kind: "add", text: "The running sum h that every layer reads (via a norm) and adds its output to. Its per-token norm and direction across layers are what the trace heatmaps show." },
  { term: "Logit lens", text: "Apply the final norm + LM head to an intermediate layer's residual stream to see which next token it would predict. Shows at which depth the answer appears." },
  { term: "MTP", kind: "mtp", text: "Multi-token prediction: an extra layer trained to predict the token after next. At inference it drafts tokens that the main model verifies (speculative decoding)." },
  { term: "Thinking mode", text: "The model writes reasoning inside <think>…</think> before the answer. Qwen3-VL ships separate Instruct/Thinking checkpoints; Qwen3.5/3.8 are one hybrid model switched with enable_thinking (3.8 adds reasoning_effort)." },
  { term: "SFT", text: "Supervised fine-tuning: maximise log-likelihood of reference assistant responses. Only assistant tokens get loss; prompt and image tokens are masked out." },
  { term: "Advantage", text: "How much better a sampled response is than expected. In GRPO-family methods: (reward − group mean) / group std, shared by every token of the response." },
  { term: "Importance ratio", text: "r = π_θ(token)/π_old(token): how much the policy being updated differs from the one that generated the rollout. Clipping or gating r keeps updates small." },
  { term: "On-policy distillation", text: "The student samples its own answers; a stronger teacher scores every token, and the student minimises KL(student‖teacher) on those tokens. Used for Qwen3 / Qwen3-VL small models." },
];

const byKind: Record<string, string> = {};
GLOSSARY.forEach((g) => { if (g.kind && !byKind[g.kind]) byKind[g.kind] = g.text; });
export const glossaryFor = (kind: string) => byKind[kind];

export const REFERENCES: { title: string; url: string; note: string }[] = [
  { title: "Qwen3-VL Technical Report", url: "https://arxiv.org/abs/2511.21631", note: "Architecture (DeepStack, interleaved MRoPE, text timestamps), 4-stage pre-training, SFT / distillation / SAPO RL." },
  { title: "Qwen3.5 model cards & configs", url: "https://huggingface.co/collections/Qwen/qwen35", note: "Hybrid Gated DeltaNet + gated attention, early fusion, MTP." },
  { title: "Qwen3.8 repository", url: "https://github.com/QwenLM/Qwen3.8", note: "Qwen3.8-27B (same architecture class as Qwen3.5), reasoning_effort, preserve_thinking." },
  { title: "Qwen3 Technical Report", url: "https://arxiv.org/abs/2505.09388", note: "Four-stage post-training, thinking-mode fusion, strong-to-weak distillation." },
  { title: "Qwen2.5-VL Technical Report", url: "https://arxiv.org/abs/2502.13923", note: "Predecessor: window-attention ViT, absolute-time MRoPE, SFT + DPO." },
  { title: "DeepSeekMath (GRPO)", url: "https://arxiv.org/abs/2402.03300", note: "Group-relative advantages, k3 KL estimator." },
  { title: "DAPO", url: "https://arxiv.org/abs/2503.14476", note: "Clip-higher, dynamic sampling, token-level loss, overlong shaping." },
  { title: "GSPO", url: "https://arxiv.org/abs/2507.18071", note: "Sequence-level ratio and clipping; stable MoE RL." },
  { title: "SAPO", url: "https://arxiv.org/abs/2511.20347", note: "Soft adaptive gate; used for Qwen3-VL." },
  { title: "Gated Delta Networks", url: "https://arxiv.org/abs/2412.06464", note: "The delta-rule linear attention used in Qwen3.5/3.8." },
  { title: "Gated Attention for LLMs", url: "https://arxiv.org/abs/2505.06708", note: "Sigmoid output gate after SDPA; attention-sink removal." },
  { title: "DeepSeek-V3 (MTP)", url: "https://arxiv.org/abs/2412.19437", note: "Multi-token prediction modules." },
  { title: "VLM-R³", url: "https://arxiv.org/abs/2505.16192", note: "Region-recognition crop loop and the VLIR dataset used for the samples and training lab." },
];
