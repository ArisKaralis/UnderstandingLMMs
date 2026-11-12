import { useMemo, useState } from "react";
import { GNode, Graph, useGraph, useSpecs, useTraces } from "../lib/api";
import { fmtBytes, fmtFlops, fmtNum } from "../lib/format";
import { useApp } from "../lib/store";
import { NodePanel } from "../components/NodePanel";
import { Disclosure, Legend, Loading } from "../components/ui";

type Override = Record<string, number>;

function editable(spec: any): { path: string; label: string; step?: number }[] {
  const f = [
    { path: "text.num_layers", label: "LLM layers" },
    { path: "text.hidden_size", label: "d_model", step: 128 },
    { path: "text.num_q_heads", label: "Query heads" },
    { path: "text.num_kv_heads", label: "KV heads" },
    { path: "text.head_dim", label: "Head dim", step: 64 },
  ];
  if (spec.text.moe) f.push({ path: "text.moe.num_experts", label: "Experts" }, { path: "text.moe.top_k", label: "Experts / token" }, { path: "text.moe.expert_ffn", label: "Expert FFN", step: 128 });
  else f.push({ path: "text.intermediate_size", label: "FFN size", step: 256 });
  if (spec.text.gdn) f.push({ path: "text.full_attention_interval", label: "1 full-attn every N layers" }, { path: "text.gdn.num_v_heads", label: "GDN V heads" });
  if (spec.vision) f.push({ path: "vision.depth", label: "ViT depth" }, { path: "vision.patch_size", label: "Patch px", step: 2 }, { path: "vision.spatial_merge_size", label: "Merge (n×n)" });
  return f;
}
const getPath = (o: any, p: string) => p.split(".").reduce((a, k) => (a == null ? a : a[k]), o);

export default function ArchitecturePage() {
  const model = useApp((s) => s.model);
  const traceId = useApp((s) => s.traceId);
  const traces = useTraces();
  const specs = useSpecs();
  const [hw, setHw] = useState<[number, number]>([448, 448]);
  const [nText, setNText] = useState(48);
  const [noImage, setNoImage] = useState(false);
  const [ov, setOv] = useState<Override>({});
  const ovKey = Object.keys(ov).length ? JSON.stringify(ov) : "";
  const base = useGraph(model, { h: hw[0], w: hw[1], n_text: nText, no_image: noImage });
  const edited = useGraph(ovKey ? model : null, { h: hw[0], w: hw[1], n_text: nText, no_image: noImage, overrides: ovKey });
  const g = (ovKey ? edited.data : base.data) ?? base.data;
  const [sel, setSel] = useState<string | null>("vision");
  const [block, setBlock] = useState<string | null>(null);
  const traceForModel = traces.data?.find((t) => t.id === traceId && t.model === model) ? traceId : null;
  const byId = useMemo(() => Object.fromEntries((g?.nodes ?? []).map((n) => [n.id, n])), [g]);
  const kids = useMemo(() => {
    const k: Record<string, GNode[]> = {};
    (g?.nodes ?? []).forEach((n) => { if (n.parent) (k[n.parent] ||= []).push(n); });
    return k;
  }, [g]);
  if (!g || !specs.data) return <Loading what="architecture" />;
  const spec = base.data!.summary!.spec;
  const fam = specs.data.models.find((m) => m.key === model);
  const llmBlocks = kids["llm"] ?? [];
  const vitBlocks = kids["vision.blocks"] ?? [];
  const blockNode = block ? byId[block] : null;
  const select = (id: string) => setSel(id);
  const node = sel ? byId[sel] ?? null : null;

  return (
    <div>
      <div className="card">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <div>
            <h1>{fam?.label} <span className="pill">{spec.architecture}</span></h1>
            <p className="small">Click a stage, then a layer, then an operation. Tensor shapes and operations are inside the closed drop-downs on the right. Concrete sizes are for the input below.</p>
          </div>
          <div className="kpi">
            <div><b>{fmtNum(g.summary!.params_total)}</b><span>params{ovKey ? " (edited)" : ""}</span></div>
            <div><b>{fmtNum(g.summary!.params_active)}</b><span>active / token</span></div>
            <div><b>{g.context.T}</b><span>sequence length T</span></div>
            <div><b>{g.context.n_vision}</b><span>vision tokens</span></div>
            <div><b>{fmtFlops(g.totals.prefill_flops)}</b><span>prefill</span></div>
          </div>
        </div>
        <div className="row">
          <label className="small">Image H <input type="number" value={hw[0]} step={32} onChange={(e) => setHw([+e.target.value, hw[1]])} style={{ width: 80 }} /></label>
          <label className="small">W <input type="number" value={hw[1]} step={32} onChange={(e) => setHw([hw[0], +e.target.value])} style={{ width: 80 }} /></label>
          <label className="small">Text tokens <input type="number" value={nText} onChange={(e) => setNText(+e.target.value)} style={{ width: 70 }} /></label>
          <label className="small"><input type="checkbox" checked={noImage} onChange={(e) => setNoImage(e.target.checked)} /> text only</label>
          {traceForModel && <span className="pill">live values from trace {traceForModel.split("__")[1]}</span>}
        </div>
        <Disclosure title={`Config playground${ovKey ? " — edited" : ""}: change the architecture and see shapes / params / cost update`}>
          <div className="grid3">
            {editable(spec).map((f) => {
              const orig = f.path === "text.full_attention_interval" ? 4 : getPath(spec, f.path);
              const v = ov[f.path] ?? orig;
              return (
                <label key={f.path} className="small row" style={{ justifyContent: "space-between" }}>
                  <span>{f.label} <span className="muted">({orig})</span></span>
                  <input type="number" value={v} step={f.step ?? 1} min={1} style={{ width: 90, borderColor: ov[f.path] !== undefined ? "var(--series-2)" : undefined }}
                    onChange={(e) => { const n = +e.target.value; const o = { ...ov }; if (n === orig || !n) delete o[f.path]; else o[f.path] = n; setOv(o); }} />
                </label>);
            })}
          </div>
          {ovKey && edited.data && <PlaygroundDiff a={base.data!} b={edited.data} />}
          {ovKey && edited.error && <div className="warn">{String(edited.error)}</div>}
          <div className="row" style={{ marginTop: 8 }}><button onClick={() => setOv({})}>Reset to real config</button>
            <span className="small muted">Weights are not involved: this is the shape/parameter/FLOP arithmetic of the architecture (same Python code the captures are validated against).</span></div>
        </Disclosure>
      </div>

      <div className="split">
        <div>
          <div className="card">
            <h3>1 · Stages</h3>
            <div className="macro">
              {["input", "vision", "embed", "positions", "llm", "final_norm", "lm_head", "mtp"].filter((id) => byId[id]).map((id, i, arr) => (
                <div key={id}>
                  <button className={`node ${sel === id ? "selected" : ""}`} onClick={() => { select(id); if (id !== "llm" && id !== "vision") setBlock(null); }}>
                    <div className="row" style={{ justifyContent: "space-between" }}>
                      <b>{byId[id].label}</b>
                      <span className="meta">{Object.entries(byId[id].outputs).map(([k, s]) => `${k} [${s.map((d) => d[1]).join("×")}]`).join("  ")}</span>
                    </div>
                    {id === "vision" && <span className="pill tag-vision">{spec.vision?.depth} ViT blocks</span>}
                    {id === "llm" && <span className="meta">{spec.text.gdn ? "3× Gated DeltaNet : 1× gated attention" : "full attention every layer"} · {spec.text.moe ? `MoE ${spec.text.moe.num_experts} experts, top-${spec.text.moe.top_k}` : "dense SwiGLU"}</span>}
                  </button>
                  {i < arr.length - 1 && <div className="flowarrow">↓</div>}
                </div>
              ))}
            </div>
          </div>

          {byId["vision"] && (
            <div className="card">
              <h3>2a · Vision encoder internals</h3>
              <div className="row" style={{ marginBottom: 8 }}>
                {(kids["vision"] ?? []).filter((n) => n.level === "op").map((n) => (
                  <button key={n.id} className={sel === n.id ? "active" : ""} onClick={() => select(n.id)}>{n.label}</button>))}
              </div>
              <div className="small muted">ViT blocks (click one); ringed = DeepStack tap feeding the LLM</div>
              <div className="stackgrid">
                {vitBlocks.map((b) => (
                  <button key={b.id} title={b.label} className={`cell vit ${block === b.id ? "selected" : ""}`} style={b.meta.deepstack_tap ? { boxShadow: "inset 0 0 0 3px var(--series-2)" } : undefined}
                    onClick={() => { setBlock(b.id); select(b.id); }}>{b.meta.index}</button>))}
              </div>
            </div>
          )}

          <div className="card">
            <h3>2b · Decoder layers</h3>
            <Legend items={[{ label: "Gated DeltaNet (linear attention)", color: "var(--series-1)" }, { label: spec.text.attn_output_gate ? "Gated full attention" : "Full attention (GQA)", color: "var(--series-2)" },
              ...(spec.text.moe ? [{ label: "MoE FFN (hatched)", color: "var(--text-muted)", pattern: true }] : []),
              ...(spec.vision?.deepstack_visual_indexes.length ? [{ label: "receives DeepStack features (ringed)", color: "var(--series-3)" }] : [])]} />
            <div className="stackgrid">
              {llmBlocks.map((b) => (
                <button key={b.id} title={b.label} className={`cell ${b.meta.mixer === "linear_attention" ? "gdn" : "attn"} ${b.meta.ffn === "moe" ? "moe" : ""} ${block === b.id ? "selected" : ""}`}
                  style={b.meta.deepstack ? { boxShadow: "inset 0 0 0 3px var(--series-3)" } : undefined}
                  onClick={() => { setBlock(b.id); select(b.id); }}>{b.meta.index}</button>))}
            </div>
          </div>

          {blockNode && (
            <div className="card">
              <h3>3 · Inside {blockNode.label}</h3>
              <div className="macro">
                {(kids[blockNode.id] ?? []).map((n) => (
                  <button key={n.id} className={`node ${sel === n.id ? "selected" : ""}`} onClick={() => select(n.id)}>
                    <div className="row" style={{ justifyContent: "space-between" }}>
                      <span><span className={`pill ${n.kind === "recurrence" ? "tag-gdn" : n.kind === "attention" ? "tag-attn" : n.kind === "experts" || n.kind === "router" ? "tag-moe" : ""}`}>{n.kind}</span> {n.label}</span>
                      <span className="meta">{n.params ? fmtNum(n.params) + " params" : ""}</span>
                    </div>
                  </button>))}
              </div>
            </div>
          )}
        </div>
        <div className="sticky"><NodePanel node={node} traceId={traceForModel} /></div>
      </div>
    </div>
  );
}

function PlaygroundDiff({ a, b }: { a: Graph; b: Graph }) {
  const rows: [string, number, number, (v: number) => string][] = [
    ["Total parameters", a.summary!.params_total, b.summary!.params_total, (v) => fmtNum(v)],
    ["Active parameters / token", a.summary!.params_active, b.summary!.params_active, (v) => fmtNum(v)],
    ["Vision parameters", a.summary!.vision.total, b.summary!.vision.total, (v) => fmtNum(v)],
    ["Prefill FLOPs", a.totals.prefill_flops, b.totals.prefill_flops, fmtFlops],
    ["KV cache / token", a.summary!.costs.kv_cache_bytes_per_token, b.summary!.costs.kv_cache_bytes_per_token, fmtBytes],
    ["GDN state (fixed)", a.summary!.costs.gdn_state_bytes_fixed, b.summary!.costs.gdn_state_bytes_fixed, fmtBytes],
    ["Vision tokens", a.context.n_vision, b.context.n_vision, String],
    ["Sequence length T", a.context.T, b.context.T, String],
  ];
  return (
    <table style={{ marginTop: 10 }}><thead><tr><th>Quantity</th><th className="num">real model</th><th className="num">edited</th><th className="num">change</th></tr></thead>
      <tbody>{rows.map(([l, x, y, f]) => (
        <tr key={l} className="diff"><td>{l}</td><td className="num">{f(x)}</td><td className={`num ${x !== y ? "changed" : ""}`}>{f(y)}</td><td className="num">{x ? `${(((y - x) / x) * 100).toFixed(1)}%` : "–"}</td></tr>))}</tbody></table>
  );
}
