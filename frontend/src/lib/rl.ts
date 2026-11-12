/** Browser-side versions of lmmviz/train_lab/losses.py for the algorithm playground (same equations). */

export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export type Algo = "grpo" | "dapo" | "gspo" | "sapo";
export type Params = { epsLow: number; epsHigh: number; tauPos: number; tauNeg: number; gspoLow: number; gspoHigh: number };
export const defaults: Params = { epsLow: 0.2, epsHigh: 0.28, tauPos: 1.0, tauNeg: 1.05, gspoLow: 3e-4, gspoHigh: 4e-4 };

/** per-token surrogate value f(r)·A as a function of the ratio r, for advantage sign `A` (±1) */
export function surrogate(algo: Algo, r: number, A: number, p: Params): number {
  switch (algo) {
    case "grpo": { const c = Math.min(Math.max(r, 1 - p.epsLow), 1 + p.epsLow); return Math.min(r * A, c * A); }
    case "dapo": { const c = Math.min(Math.max(r, 1 - p.epsLow), 1 + p.epsHigh); return Math.min(r * A, c * A); }
    case "gspo": { const c = Math.min(Math.max(r, 1 - p.gspoLow), 1 + p.gspoHigh); return Math.min(r * A, c * A); }
    case "sapo": { const t = A > 0 ? p.tauPos : p.tauNeg; return (4 / t) * sigmoid(t * (r - 1)) * A; }
  }
}

/** weight multiplying r·∇log π in the gradient (1 = full policy gradient, 0 = no gradient) */
export function gradWeight(algo: Algo, r: number, A: number, p: Params): number {
  if (algo === "sapo") { const t = A > 0 ? p.tauPos : p.tauNeg; const s = sigmoid(t * (r - 1)); return 4 * s * (1 - s); }
  const [lo, hi] = algo === "grpo" ? [p.epsLow, p.epsLow] : algo === "dapo" ? [p.epsLow, p.epsHigh] : [p.gspoLow, p.gspoHigh];
  if (A > 0) return r > 1 + hi ? 0 : 1;
  return r < 1 - lo ? 0 : 1;
}

export function groupAdvantages(rewards: number[]): number[] {
  const m = rewards.reduce((a, b) => a + b, 0) / rewards.length;
  const sd = Math.sqrt(rewards.reduce((a, b) => a + (b - m) ** 2, 0) / rewards.length);
  return rewards.map((r) => (r - m) / (sd + 1e-6));
}

export const ALGO_INFO: Record<Algo, { name: string; eq: string; text: string; ref: string }> = {
  grpo: {
    name: "GRPO",
    eq: String.raw`\mathcal J=\frac1G\sum_i\frac1{|o_i|}\sum_t\min\!\big(r_{i,t}\hat A_i,\ \mathrm{clip}(r_{i,t},1\!-\!\varepsilon,1\!+\!\varepsilon)\hat A_i\big)-\beta\,\mathbb D_{KL}`,
    text: "Group Relative Policy Optimisation. No value model: sample G answers per prompt, the advantage of each is its reward standardised within the group. PPO-style clipping per token, plus a k3 KL penalty to a reference model. Used by Qwen3 reasoning RL.",
    ref: "DeepSeekMath, arXiv 2402.03300",
  },
  dapo: {
    name: "DAPO",
    eq: String.raw`\mathcal J=\frac{1}{\sum_i|o_i|}\sum_i\sum_t\min\!\big(r_{i,t}\hat A_i,\ \mathrm{clip}(r_{i,t},1\!-\!\varepsilon_{low},1\!+\!\varepsilon_{high})\hat A_i\big)`,
    text: "GRPO with four fixes: clip-higher (ε_high=0.28 lets rare tokens grow → avoids entropy collapse), dynamic sampling (drop groups where all answers are right or all wrong), token-level averaging (long answers count per token), soft overlong penalty. No KL.",
    ref: "arXiv 2503.14476",
  },
  gspo: {
    name: "GSPO",
    eq: String.raw`s_i=\Big(\frac{\pi_\theta(y_i|x)}{\pi_{old}(y_i|x)}\Big)^{1/|y_i|},\quad \mathcal J=\frac1G\sum_i\min\!\big(s_i\hat A_i,\ \mathrm{clip}(s_i,1\!-\!\varepsilon,1\!+\!\varepsilon)\hat A_i\big)`,
    text: "Group Sequence Policy Optimisation (Qwen). The importance ratio and the clipping act on the whole response (length-normalised geometric mean), matching the sequence-level reward. Much more stable for MoE models, where expert routing changes between updates make token ratios noisy. Tiny clip range (3e-4 / 4e-4).",
    ref: "arXiv 2507.18071",
  },
  sapo: {
    name: "SAPO",
    eq: String.raw`\mathcal J=\frac1G\sum_i\frac1{|y_i|}\sum_t f_{i,t}(r_{i,t})\hat A_i,\quad f(x)=\tfrac{4}{\tau}\sigma\big(\tau(x-1)\big),\ \tau=\begin{cases}\tau_{pos}&\hat A>0\\\tau_{neg}&\text{else}\end{cases}`,
    text: "Soft Adaptive Policy Optimisation (Qwen) — the algorithm used for Qwen3-VL reasoning RL. Replaces the hard clip by a smooth sigmoid gate: the gradient weight sech²(τ(r−1)/2) is 1 on-policy and decays smoothly, faster for negative advantages (τ_neg > τ_pos).",
    ref: "arXiv 2511.20347",
  },
};
