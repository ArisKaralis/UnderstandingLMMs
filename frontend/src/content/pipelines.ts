/** Training pipelines as published. "status" marks what the sources actually disclose. */
export type Stage = {
  id: string; name: string; phase: "pre" | "post"; status: "published" | "vendor-claim" | "inherited";
  trains: string; data: string; tokens?: string; seq?: string; objective: string; eq?: string; io: string; source: string;
};

export const PIPELINES: Record<string, { label: string; stages: Stage[]; notes: string[] }> = {
  qwen3_vl: {
    label: "Qwen3-VL",
    notes: [
      "Thinking and non-thinking (Instruct) variants are post-trained separately.",
      "Pre-training loss uses square-root-normalised per-token weighting (instead of per-sample) to balance text and multimodal data.",
    ],
    stages: [
      { id: "s0", name: "S0 · Vision–language alignment", phase: "pre", status: "published", trains: "Merger only (ViT + LLM frozen)", data: "Image–caption pairs, visual knowledge, OCR", tokens: "~67B", seq: "8,192",
        objective: "Next-token cross-entropy on text tokens", eq: String.raw`\mathcal L=-\sum_{t\in\text{text}}\log p_\theta(x_t\mid x_{<t},\text{img})`,
        io: "Input: image + caption prompt. Output: caption tokens. Only the 2-layer MLP merger learns to map ViT features into the LLM's embedding space.", source: "Qwen3-VL report, Table 1" },
      { id: "s1", name: "S1 · Multimodal pre-training", phase: "pre", status: "published", trains: "All parameters", data: "Interleaved image–text documents, grounding, VQA, STEM, some video, plus text-only data", tokens: "~1T", seq: "8,192",
        objective: "Next-token cross-entropy, sqrt-normalised per-token weighting", io: "Long interleaved web documents: text tokens predicted given all preceding text and images.", source: "Qwen3-VL report §3" },
      { id: "s2", name: "S2 · Long-context pre-training", phase: "pre", status: "published", trains: "All parameters", data: "More text-only share; much more video and agent instruction data", tokens: "~1T", seq: "32,768",
        objective: "Next-token cross-entropy", io: "Multi-page documents, long videos (frames + text timestamps), GUI trajectories.", source: "Qwen3-VL report §3" },
      { id: "s3", name: "S3 · Ultra-long-context adaptation", phase: "pre", status: "published", trains: "All parameters", data: "Long video and long documents", tokens: "100B", seq: "262,144",
        objective: "Next-token cross-entropy", io: "Book-length interleaved inputs, hour-long video.", source: "Qwen3-VL report Table 1" },
      { id: "sft", name: "SFT", phase: "post", status: "published", trains: "All parameters", data: "~1.2M samples: ⅓ text, ⅔ image/video-text; single/multi-turn, multi-image, agentic (image search)", seq: "1 epoch at 32K, then 1 epoch at 256K",
        objective: "Cross-entropy on assistant tokens only", eq: String.raw`\mathcal L_{SFT}=-\frac{1}{|\mathcal A|}\sum_{t\in\mathcal A}\log \pi_\theta(y_t\mid x,y_{<t})`,
        io: "Input: chat-formatted conversation with images. Labels: assistant turns (thinking models: CoT format). Prompt and image tokens are masked (label = −100). Query filtering and reward-model response filtering beforehand.", source: "Qwen3-VL report §4.2" },
      { id: "cot", name: "Long-CoT cold start", phase: "post", status: "published", trains: "All parameters", data: "VL : text ≈ 1:1; VQA, OCR, grounding, video, STEM, agentic. Keep hard items (low pass rate / long responses); drop VL-math items solvable without the image",
        objective: "SFT on long chain-of-thought responses", io: "Input: hard multimodal problem. Label: long verified reasoning trace + answer.", source: "Qwen3-VL report §4.2.2" },
      { id: "distill", name: "Strong-to-weak distillation", phase: "post", status: "published", trains: "Student LLM (text-only data)", data: "Teacher outputs (off-policy), then student samples scored by teacher (on-policy)",
        objective: "Off-policy: SFT on teacher outputs. On-policy: minimise KL between student and teacher logits on student samples", eq: String.raw`\mathcal L=\mathbb E_{y\sim\pi_s}\sum_t \mathrm{KL}\big(\pi_s(\cdot\mid y_{<t})\,\|\,\pi_T(\cdot\mid y_{<t})\big)`,
        io: "Student generates; teacher's full next-token distribution at every student position is the target. Used for 2B/4B/8B.", source: "Qwen3-VL report §4.3; Qwen3 report" },
      { id: "rl1", name: "Reasoning RL (SAPO)", phase: "post", status: "published", trains: "All parameters", data: "~30K verifiable queries: math, code, logic, visual grounding, puzzles; 16 rollouts/query; drop pass-rate > 90%",
        objective: "SAPO (soft adaptive policy optimisation) on rule/code-verified rewards; language-consistency penalty, no explicit format reward",
        io: "Prompt → 16 sampled responses → verifier reward per response → group-standardised advantages → soft-gated policy-gradient update.", source: "Qwen3-VL report §4.4.1; SAPO arXiv 2511.20347" },
      { id: "rl2", name: "General RL", phase: "post", status: "published", trains: "All parameters", data: "VQA, captioning, OCR, document parsing, grounding, clock reading; instruction following; preference",
        objective: "Hybrid rewards: rules (format, IF constraints) + model judges (Qwen2.5-VL-72B / Qwen3) vs references; targeted 'corrective' data (counting, clocks, repetition, language mixing)",
        io: "Prompt → sampled answers → rule + judge rewards → update.", source: "Qwen3-VL report §4.4.2" },
      { id: "twi", name: "Thinking with images (agentic RL)", phase: "post", status: "published", trains: "All parameters", data: "~10K grounding cold-start → ~120K multi-turn tool interactions distilled from Qwen2.5-VL-32B agents",
        objective: "Multi-turn tool-integrated RL with 3 rewards: answer accuracy (Qwen3-32B judge), multi-turn reasoning (Qwen2.5-VL-72B judge), tool-call count vs expert target",
        io: "Think → call crop/zoom tool → observe new image → … → answer. Same loop as VLM-R³ used in your experiments.", source: "Qwen3-VL report §4.5" },
    ],
  },
  qwen3_5: {
    label: "Qwen3.5 / Qwen3.8",
    notes: [
      "Qwen has not published a technical report with post-training algorithms for Qwen3.5 or Qwen3.8. Cards below quote the model cards; methods are marked 'vendor claim'.",
      "Qwen3.8-27B uses the Qwen3.5 architecture class and builds on Qwen3.6-27B.",
    ],
    stages: [
      { id: "pt", name: "Early-fusion multimodal pre-training", phase: "pre", status: "vendor-claim", trains: "All parameters (incl. MTP head, multi-step)", data: "'Trillions of multimodal tokens', 201 languages",
        objective: "Next-token CE + MTP loss", eq: String.raw`\mathcal L=\mathcal L_{CE}+\frac{\lambda}{D}\sum_{k=1}^{D}\mathcal L^{k}_{MTP}`, io: "Text, images and video in one sequence from the first stage.", source: "Qwen3.5 model card" },
      { id: "post", name: "Post-training (SFT + RL)", phase: "post", status: "vendor-claim", trains: "All parameters", data: "Not disclosed",
        objective: "'RL scaled across million-agent environments with progressively complex task distributions'; asynchronous RL framework",
        io: "Hybrid thinking model: one checkpoint serves thinking (<think>…</think>) and non-thinking modes via enable_thinking.", source: "Qwen3.5-397B model card" },
      { id: "q38", name: "Qwen3.8 additions", phase: "post", status: "vendor-claim", trains: "—", data: "—",
        objective: "reasoning_effort levels (xhigh / medium / low) and preserve_thinking (keep reasoning from earlier turns) exposed at inference; method not disclosed",
        io: "Thinking sampling: T=1.0, top-p 0.95, top-k 20. Non-thinking: T=0.7, top-p 0.8, presence penalty 1.5.", source: "Qwen3.8-27B model card" },
      { id: "inh", name: "Inherited Qwen3 recipe (for reference)", phase: "post", status: "inherited", trains: "All parameters", data: "Long-CoT cold start → reasoning RL (GRPO, 3,995 queries) → thinking-mode fusion → general RL (20+ tasks)",
        objective: "GRPO; fusion SFT mixing <think>…</think> and empty-think responses; rule / reference-judge / reward-model rewards",
        io: "This is the published Qwen3 text recipe that the hybrid-thinking design descends from.", source: "Qwen3 report arXiv 2505.09388" },
    ],
  },
};
