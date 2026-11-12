# Trace bundle format (v1)

One directory per run: `<model>__<sample>__<mode>__<intervention>/`. Written atomically (`*.partial` and then renamed). `manifest.complete = true` marks a finished bundle; the capture runner skips complete bundles, so it can be resumed.

| File | Content |
|---|---|
| `manifest.json` | model entry, sample, mode (`direct` / `r3`), intervention spec, capture policy, chat-template kwargs, prompt text, full text, T, vision-token count, timings, library versions, device, git commits (this repo + MLLM_Testing), size |
| `tokens.json` | `ids`, `strings`, `segments` (system / user / vision / vision_marker / special / assistant_prefill / thinking / answer), `n_prompt`, `images[]` (grid_thw, resized/original size, n_patches, n_tokens, token_start, patch_offset, `raster_order`), `lens_rows` |
| `graph.json` | `build_graph(spec)` with this run's concrete shapes (first image; T equals the real sequence) |
| `stats.parquet` | one row per recorded module output: node id, module name, shape, dtype, mean/std/min/max/absmax/L2, zero fraction, non-finite count, 32-bin histogram |
| `tensors/*.safetensors` + `index.json` | full fp16 tensors chosen by the policy, grouped per layer file |
| `previews/*.npy` + `index.json` | downsampled artefacts (below) |
| `outputs.json` | reply, R³ parts / rounds / boxes, per-step top-10 log-probs, rule score vs gold, `lens_vocab` |
| `images/<k>.png` | the input image and any crops fed to the model |

## Previews

| Key | Shape | Meaning |
|---|---|---|
| `position_ids` | [3, T] | MRoPE (t, h, w) per token |
| `residual_norm`, `residual_rel_update`, `residual_cos_prev` | [L, T] | per-layer residual stream statistics |
| `residual_pca3` | [L, T, 3] | per-layer PCA projection |
| `logit_lens_ids`, `logit_lens_probs` | [L, R, 5] | top-5 next-token ids/probabilities per layer at `lens_rows` |
| `logit_lens_target_p` | [L, R] | probability of the true next token |
| `token_logprob`, `token_entropy` | [T] | teacher-forced |
| `attn.<l>.sink` / `.entropy` | [H] | mean attention to token 0; mean entropy |
| `attn.<l>.last_row` | [H, T] | last prompt token → all keys |
| `attn.<l>.answer_to_image` | [H, N_vis] | answer positions → vision tokens (mean) |
| `attn.<l>.mean` | [T, T] | head-mean attention (T ≤ policy.max_attn_tokens) |
| `attn_gate_mean.<l>` | [T, H] | gated-attention sigmoid gate, mean over head dims (Qwen3.5/3.8) |
| `gdn.<l>.alpha` / `.beta` | [T, H_v] | Gated DeltaNet decay / write strength |
| `gdn.<l>.state_norm` / `.write_norm` | [T, H_v] | ‖S_t‖_F and ‖β(v − Sᵀk)‖ from an fp32 re-run of the recurrence |
| `moe.<l>.experts` / `.weights` | [T, k] | routing choices and weights |
| `moe.<l>.load` / `.entropy` | [E] / [T] | tokens per expert; router entropy |
| `vit_attn_received.<i>` | [H, N_p] | attention each patch receives (ViT sequence order; map with `raster_order`) |
| `vit_attn_mean.<i>` | [N_p, N_p] | head-mean ViT attention for the first image (N_p ≤ 2304) |
| `vit_residual_norm` | [D, N_p] | per-block patch norms |

## Policies

| Policy | Full tensors | Attention layers | Typical size |
|---|---|---|---|
| `lite` | none | ViT taps; every 4th LLM layer | 5–40 MB |
| `standard` | embedding, vision path, DeepStack, first/mid/last 2 layers, GDN final states | all | 20–150 MB (≤2B), more for larger models |
| `full` | every mapped module output | all | small models only |

Validate with `python scripts/local/validate_bundle.py <bundles…>`. It checks that the captured shapes agree with `graph.json`.
