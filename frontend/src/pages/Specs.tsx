import { useMemo, useState } from "react";
import { ModelRow, useSpecs } from "../lib/api";
import { fmtBytes, fmtNum } from "../lib/format";
import { Loading } from "../components/ui";
import { useApp } from "../lib/store";

type Col = { key: string; label: string; get: (m: ModelRow) => string | number; num?: boolean; src?: string };

const COLS: Col[] = [
  { key: "params", label: "Total params", get: (m) => m.spec.params.total_with_mtp, num: true, src: "derived (matches HF safetensors exactly)" },
  { key: "active", label: "Active / token", get: (m) => m.spec.params.active, num: true },
  { key: "layers", label: "Layers", get: (m) => m.spec.text.num_layers, num: true, src: "text_config.num_hidden_layers" },
  { key: "mix", label: "Mixer layout", get: (m) => { const lt = m.spec.text.layer_types; const f = lt.filter((x) => x === "full_attention").length; return f === lt.length ? "full attn" : `${lt.length - f} GDN + ${f} attn`; }, src: "text_config.layer_types" },
  { key: "d", label: "d_model", get: (m) => m.spec.text.hidden_size, num: true, src: "hidden_size" },
  { key: "ffn", label: "FFN", get: (m) => m.spec.text.moe ? `MoE ${m.spec.text.moe.num_experts}×${m.spec.text.moe.expert_ffn} top-${m.spec.text.moe.top_k}${m.spec.text.moe.shared_ffn ? " +shared" : ""}` : String(m.spec.text.intermediate_size), src: "intermediate_size / num_experts / moe_intermediate_size" },
  { key: "heads", label: "Attn Q/KV × d_h", get: (m) => `${m.spec.text.num_q_heads}/${m.spec.text.num_kv_heads} × ${m.spec.text.head_dim}`, src: "num_attention_heads / num_key_value_heads / head_dim" },
  { key: "gate", label: "Attn gate", get: (m) => (m.spec.text.attn_output_gate ? "sigmoid" : "–"), src: "attn_output_gate" },
  { key: "gdn", label: "GDN QK/V heads", get: (m) => (m.spec.text.gdn ? `${m.spec.text.gdn.num_k_heads}/${m.spec.text.gdn.num_v_heads} × ${m.spec.text.gdn.k_head_dim}` : "–"), src: "linear_num_key_heads / linear_num_value_heads" },
  { key: "rope", label: "RoPE (θ, rotary, mrope)", get: (m) => `${(m.spec.text.rope.theta / 1e6).toFixed(0)}M, ${m.spec.text.rotary_dim}/${m.spec.text.head_dim}, [${m.spec.text.rope.mrope_section}]`, src: "rope_parameters" },
  { key: "vocab", label: "Vocab", get: (m) => m.spec.text.vocab_size, num: true },
  { key: "ctx", label: "Context", get: (m) => m.spec.text.max_position_embeddings, num: true },
  { key: "vit", label: "ViT depth/width/heads", get: (m) => (m.spec.vision ? `${m.spec.vision.depth}/${m.spec.vision.hidden_size}/${m.spec.vision.num_heads}` : "–"), src: "vision_config" },
  { key: "patch", label: "Patch / merge", get: (m) => (m.spec.vision ? `${m.spec.vision.patch_size}px / ${m.spec.vision.spatial_merge_size}×${m.spec.vision.spatial_merge_size}` : "–") },
  { key: "ds", label: "DeepStack ViT layers", get: (m) => (m.spec.vision?.deepstack_visual_indexes.length ? m.spec.vision.deepstack_visual_indexes.join(",") : "–"), src: "deepstack_visual_indexes" },
  { key: "mtp", label: "MTP", get: (m) => m.spec.text.mtp_layers || "–" },
  { key: "tied", label: "Tied emb.", get: (m) => (m.spec.text.tie_word_embeddings ? "yes" : "no") },
  { key: "kv", label: "KV cache / token", get: (m) => m.spec.costs.kv_cache_bytes_per_token, num: true },
  { key: "state", label: "GDN state (fixed)", get: (m) => m.spec.costs.gdn_state_bytes_fixed, num: true },
  { key: "vt", label: "Vision tokens @1MP", get: (m) => m.spec.costs.vision_tokens_1mp?.llm_tokens ?? "–", num: true },
];

const fmtCell = (c: Col, v: string | number) =>
  typeof v !== "number" ? v : c.key === "kv" || c.key === "state" ? (v ? fmtBytes(v) : "–") : c.key === "params" || c.key === "active" ? fmtNum(v) : v.toLocaleString();

export default function SpecsPage() {
  const specs = useSpecs();
  const setModel = useApp((s) => s.setModel);
  const [cmp, setCmp] = useState<string[]>(["qwen3-vl-8b", "qwen3.5-9b", "qwen3.8-27b"]);
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(null);
  const models = useMemo(() => {
    const ms = [...(specs.data?.models ?? [])];
    if (sort) { const c = COLS.find((x) => x.key === sort.key)!; ms.sort((a, b) => (c.get(a) > c.get(b) ? 1 : -1) * sort.dir); }
    return ms;
  }, [specs.data, sort]);
  if (!specs.data) return <Loading what="specifications" />;
  const chosen = cmp.map((k) => specs.data!.models.find((m) => m.key === k)!).filter(Boolean);
  return (
    <div>
      <div className="card">
        <h1>Model specifications</h1>
        <p>All open Qwen-VL-family vision-language models up to 100B parameters. Every number is read from the model's <code>config.json</code> on Hugging Face and derived by <code>lmmviz/specs/derive.py</code>; derived parameter totals match the published safetensors sizes exactly. Click a row to make it the active model; tick up to 4 to compare.</p>
        <div style={{ overflowX: "auto" }}>
          <table>
            <thead><tr><th></th><th>Model</th>{COLS.map((c) => (
              <th key={c.key} className={c.num ? "num" : ""} title={c.src} onClick={() => setSort({ key: c.key, dir: sort?.key === c.key ? (-sort.dir as 1 | -1) : 1 })}>{c.label}{sort?.key === c.key ? (sort.dir > 0 ? " ▲" : " ▼") : ""}</th>))}</tr></thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.key}>
                  <td><input type="checkbox" checked={cmp.includes(m.key)} onChange={(e) => setCmp(e.target.checked ? [...cmp, m.key].slice(-4) : cmp.filter((k) => k !== m.key))} /></td>
                  <td><a href="#" onClick={(e) => { e.preventDefault(); setModel(m.key); }}>{m.label}</a> <span className="pill">{specs.data!.families[m.family].label}</span></td>
                  {COLS.map((c) => <td key={c.key} className={c.num ? "num" : ""}>{fmtCell(c, c.get(m))}</td>)}
                </tr>))}
            </tbody>
          </table>
        </div>
      </div>
      {chosen.length > 1 && (
        <div className="card">
          <h2>Compare</h2>
          <p className="small">Highlighted rows differ between the selected models.</p>
          <table>
            <thead><tr><th>Field</th>{chosen.map((m) => <th key={m.key}>{m.label}</th>)}</tr></thead>
            <tbody>{COLS.map((c) => {
              const vals = chosen.map((m) => fmtCell(c, c.get(m)));
              const diff = new Set(vals).size > 1;
              return (<tr key={c.key} className="diff"><td title={c.src}>{c.label}</td>{vals.map((v, i) => <td key={i} className={diff ? "changed" : ""}>{v}</td>)}</tr>);
            })}</tbody>
          </table>
        </div>
      )}
      <div className="card">
        <h2>Lineage: what changed, when</h2>
        <table><tbody>{specs.data.lineage.map((l) => (<tr key={l.name}><td className="muted">{l.date}</td><td><b>{l.name}</b></td><td style={{ whiteSpace: "normal" }}>{l.change}</td></tr>))}</tbody></table>
      </div>
      <div className="card">
        <h2>Generation defaults (from generation_config.json)</h2>
        <table><thead><tr><th>Model</th><th>temperature</th><th>top_p</th><th>top_k</th><th>max pixels (processor)</th></tr></thead>
          <tbody>{specs.data.models.map((m) => (<tr key={m.key}><td>{m.label}</td><td>{m.spec.generation?.temperature ?? "–"}</td><td>{m.spec.generation?.top_p ?? "–"}</td><td>{m.spec.generation?.top_k ?? "–"}</td>
            <td>{m.spec.preprocessor?.size?.longest_edge?.toLocaleString() ?? m.spec.preprocessor?.max_pixels?.toLocaleString() ?? "–"}</td></tr>))}</tbody></table>
      </div>
    </div>
  );
}
