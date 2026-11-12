import { useEffect, useMemo, useState } from "react";
import { Arr, rowOf, usePreview, useTracePart, useTraces, TraceRow } from "../lib/api";
import { fmtNum, showTok } from "../lib/format";
import { useApp } from "../lib/store";
import { Bars, Heatmap, LineChart } from "../components/charts";
import { ImageHeat, mergedToRaster, toRaster } from "../components/ImageHeat";
import { Disclosure, Empty, Legend, Loading, TeX, useTip } from "../components/ui";

const STAGES = ["1 · Input", "2 · Tokens & positions", "3 · Vision encoder", "4 · Merger & fusion", "5 · Decoder layers", "6 · Logit lens", "7 · Output"];

export function TracePicker({ value, onChange, rows, label = "Trace" }: { value: string | null; onChange: (id: string) => void; rows: TraceRow[]; label?: string }) {
  return (
    <label className="small row">{label}
      <select value={value ?? ""} onChange={(e) => onChange(e.target.value)} style={{ maxWidth: 520 }}>
        <option value="" disabled>choose a captured run…</option>
        {rows.map((t) => <option key={t.id} value={t.id}>{t.model} · {t.sample} · {t.mode} · {t.intervention} ({t.T} tok{t.demo ? ", demo" : ""})</option>)}
      </select>
    </label>
  );
}

export default function TracePage() {
  const { traceId, setTrace, model, setModel } = useApp();
  const traces = useTraces();
  const [stage, setStage] = useState(0);
  const rows = traces.data ?? [];
  useEffect(() => {
    if (!rows.length) return;
    const cur = rows.find((r) => r.id === traceId);
    if (!cur) { const pick = rows.find((r) => r.model === model) ?? rows[0]; setTrace(pick.id); }
  }, [rows.length]);
  const manifest = useTracePart(traceId, "manifest");
  const tokens = useTracePart(traceId, "tokens");
  const outputs = useTracePart(traceId, "outputs");
  const pidx = useTracePart<Record<string, { shape: number[] }>>(traceId, "previews");
  if (traces.isLoading) return <Loading what="traces" />;
  if (!rows.length) return <Empty>No trace bundles yet. Run <code>python -m lmmviz.capture.run …</code> locally or on the HPC and copy them to <code>data/traces/</code>.</Empty>;
  const ready = manifest.data && tokens.data && outputs.data && pidx.data;
  return (
    <div>
      <div className="card">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <TracePicker rows={rows} value={traceId} onChange={(id) => { setTrace(id); const r = rows.find((x) => x.id === id); if (r) setModel(r.model); }} />
          {manifest.data && <span className="small muted">{manifest.data.model.label} · policy {manifest.data.policy.name} · {manifest.data.env.device} · forward {manifest.data.timing.trace_forward_s.toFixed(1)}s{manifest.data.demo ? " · DEMO (random weights)" : ""}</span>}
        </div>
        {manifest.data?.demo && <div className="warn small">Demo bundle from a tiny random-weight model: shapes and plumbing are real, values are meaningless. Real captures come from the HPC.</div>}
        <div className="stagenav" style={{ marginTop: 10 }}>
          {STAGES.map((s, i) => <button key={s} className={stage === i ? "active" : ""} onClick={() => setStage(i)}>{s}</button>)}
          <button onClick={() => setStage(Math.max(0, stage - 1))}>← prev</button><button onClick={() => setStage(Math.min(STAGES.length - 1, stage + 1))}>next →</button>
        </div>
      </div>
      {!ready ? <Loading what="trace" /> : (
        <TraceStage stage={stage} tid={traceId!} m={manifest.data} tok={tokens.data} out={outputs.data} pidx={pidx.data!} />
      )}
    </div>
  );
}

function TraceStage({ stage, tid, m, tok, out, pidx }: { stage: number; tid: string; m: any; tok: any; out: any; pidx: Record<string, { shape: number[] }> }) {
  switch (stage) {
    case 0: return <InputStage tid={tid} m={m} tok={tok} out={out} />;
    case 1: return <TokenStage tid={tid} tok={tok} />;
    case 2: return <VisionStage tid={tid} tok={tok} pidx={pidx} m={m} />;
    case 3: return <MergerStage tid={tid} tok={tok} m={m} />;
    case 4: return <LayerStage tid={tid} tok={tok} pidx={pidx} m={m} />;
    case 5: return <LensStage tid={tid} tok={tok} out={out} />;
    default: return <OutputStage tid={tid} tok={tok} out={out} m={m} />;
  }
}

/* ---------------------------------------------------------------- 1 input */
function InputStage({ tid, m, tok, out }: any) {
  const [grid, setGrid] = useState(true);
  return (
    <div className="grid2">
      <div className="card">
        <h2>Question</h2>
        <p style={{ whiteSpace: "pre-wrap", color: "var(--text-primary)" }}>{m.sample.question}</p>
        <div className="small muted">Sample <b>{m.sample.id}</b> ({m.sample.source}) · gold: {m.sample.answers.slice(0, 3).join(" / ")} · mode <b>{m.mode}</b>{m.mode === "r3" ? " (VLM-R³ crop loop, as in MLLM_Testing)" : ""}</div>
        <Disclosure title="Full prompt after the chat template">
          <pre style={{ whiteSpace: "pre-wrap" }}>{m.prompt_text}</pre>
          <p className="small">Template kwargs: <code>{JSON.stringify(m.template_kwargs)}</code>. The single <code>&lt;|image_pad|&gt;</code> is expanded by the processor to one placeholder per visual token.</p>
        </Disclosure>
        <Disclosure title="Intervention for this run"><pre>{JSON.stringify(m.intervention, null, 1)}</pre></Disclosure>
      </div>
      <div className="card">
        <h2>Images → patches → tokens</h2>
        <label className="small"><input type="checkbox" checked={grid} onChange={(e) => setGrid(e.target.checked)} /> show 16 px patch grid and 2×2 merge windows (orange)</label>
        {tok.images.map((im: any) => (
          <div key={im.index} style={{ margin: "10px 0" }}>
            <div className="small"><b>{im.index === 0 ? "Input image" : `Crop ${im.index} (round ${im.index + 1})`}</b> · original {im.original[1]}×{im.original[0]} → resized {im.resized[1]}×{im.resized[0]} · grid t×h×w = {im.grid_thw.join("×")} = {im.n_patches} patches → <b>{im.n_tokens} LLM tokens</b></div>
            <ImageHeat src={`/api/traces/${tid}/image/${im.index}`} gh={im.grid_thw[1]} gw={im.grid_thw[2]} showGrid={grid} mergeGrid={grid ? 2 : 0} width={Math.min(520, 18 * im.grid_thw[2])} />
          </div>))}
        <Disclosure title="Tensor details: pixel_values">
          {tok.images.map((im: any) => (
            <pre key={im.index}>{`image ${im.index}: [${im.resized[0]}, ${im.resized[1]}, 3] uint8
→ normalise (x/255 − 0.5)/0.5, repeat to T=2 frames
→ pixel_values [${im.n_patches}, 3·2·16·16 = 1536]   grid_thw = [${im.grid_thw.join(", ")}]`}</pre>))}
        </Disclosure>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- 2 tokens */
function TokenStage({ tid, tok }: any) {
  const pos = usePreview(tid, "position_ids");
  const [hl, setHl] = useState<number | null>(null);
  const { tip, show, hide } = useTip();
  const segCounts = useMemo(() => tok.segments.reduce((a: any, s: string) => ((a[s] = (a[s] || 0) + 1), a), {}), [tok]);
  const P = pos.data;
  const T = tok.ids.length;
  const posOf = (i: number) => (P ? [P.data[i], P.data[T + i], P.data[2 * T + i]] : null);
  // collapse long vision runs for readability
  const items: { i: number; n?: number }[] = [];
  for (let i = 0; i < T; i++) {
    if (tok.segments[i] === "vision" && i > 0 && tok.segments[i - 1] === "vision") { items[items.length - 1].n!++; continue; }
    items.push(tok.segments[i] === "vision" ? { i, n: 1 } : { i });
  }
  return (
    <div>
      <div className="card">
        <h2>Token sequence (T = {T})</h2>
        <Legend items={[["user", "var(--surface-2)"], ["vision", "color-mix(in srgb, var(--series-3) 30%, transparent)"], ["answer", "color-mix(in srgb, var(--series-1) 22%, transparent)"], ["thinking", "color-mix(in srgb, var(--series-7) 18%, transparent)"]].map(([l, c]) => ({ label: `${l} (${segCounts[l] ?? 0})`, color: c }))} />
        <div className="tokstrip">
          {items.map(({ i, n }) => n ? (
            <span key={i} className="tok vision" onMouseMove={(e) => show(e, <span>{n} image tokens (positions {i}–{i + n - 1})<br />each = one 2×2 window of 16 px patches</span>)} onMouseLeave={hide}>▦ ×{n}</span>
          ) : (
            <span key={i} className={`tok ${tok.segments[i]} ${hl === i ? "hl" : ""}`} onClick={() => setHl(i)}
              onMouseMove={(e) => { const p = posOf(i); show(e, <span>#{i} id {tok.ids[i]} <b>{JSON.stringify(tok.strings[i])}</b><br />segment {tok.segments[i]}{p ? <><br />MRoPE (t,h,w) = ({p.join(", ")})</> : null}</span>); }} onMouseLeave={hide}>{showTok(tok.strings[i])}</span>
          ))}
        </div>
        {tip}
      </div>
      {P && (
        <div className="card">
          <h2>3-D positions (MRoPE)</h2>
          <p>Text tokens advance t, h and w together. Image tokens share one t and use their row (h) and column (w) in the merged grid. The text after an image continues from max(position) + 1, which is why positions can be smaller than token indices.</p>
          <LineChart width={1000} height={220} xLabel="token index" yLabel="position id"
            series={[{ name: "t", values: rowOf(P, 0), color: "var(--series-1)" }, { name: "h", values: rowOf(P, 1), color: "var(--series-2)" }, { name: "w", values: rowOf(P, 2), color: "var(--series-3)" }]}
            marks={tok.images.filter((im: any) => im.token_start != null).map((im: any) => ({ x: im.token_start, label: `img ${im.index}` }))} />
          <Disclosure title="Operation: how (t,h,w) become rotations">
            <TeX block tex={String.raw`q'_{[2j,2j+1]} = \begin{pmatrix}\cos\theta_j p_{a(j)} & -\sin\theta_j p_{a(j)}\\ \sin\theta_j p_{a(j)} & \cos\theta_j p_{a(j)}\end{pmatrix} q_{[2j,2j+1]},\quad \theta_j=\theta^{-2j/d_{rot}}`} />
            <p className="small">Interleaved: frequency index j uses axis a(j) = h if j mod 3 = 1, w if j mod 3 = 2, t otherwise (within the mrope_section counts). Position-id tensor shape: [3, {T}].</p>
          </Disclosure>
        </div>
      )}
    </div>
  );
}

/* ---------------------------------------------------------------- 3 vision */
function VisionStage({ tid, tok, pidx, m }: any) {
  const layers = useMemo(() => Object.keys(pidx).filter((k) => k.startsWith("vit_attn_received.")).map((k) => +k.split(".")[1]).sort((a, b) => a - b), [pidx]);
  const [layer, setLayer] = useState<number>(layers[layers.length - 1] ?? 0);
  const [head, setHead] = useState<number>(-1);
  const [img, setImg] = useState(0);
  const [qpatch, setQpatch] = useState<[number, number] | null>(null);
  const recv = usePreview(tid, `vit_attn_received.${layer}`);
  const ent = usePreview(tid, `vit_attn_entropy.${layer}`);
  const hasMean = !!pidx[`vit_attn_mean.${layer}`];
  const im = tok.images[img];
  const seqIndex = qpatch && im ? im.raster_order.indexOf(qpatch[0] * im.grid_thw[2] + qpatch[1]) : null;
  const qrow = usePreview(hasMean && seqIndex !== null && seqIndex >= 0 && img === 0 ? tid : null, `vit_attn_mean.${layer}`, { row: seqIndex ?? 0 });
  const vnorm = usePreview(tid, "vit_residual_norm");
  if (!layers.length || !im) return <Empty>No vision attention recorded in this bundle.</Empty>;
  const H = recv.data?.shape[0] ?? 0, N = recv.data?.shape[1] ?? 0;
  let seq: number[] | null = null;
  if (recv.data) {
    if (head < 0) { seq = new Array(N).fill(0); for (let h = 0; h < H; h++) for (let n = 0; n < N; n++) seq[n] += recv.data.data[h * N + n] / H; }
    else seq = rowOf(recv.data, head);
  }
  const overlay = qrow.data && img === 0 ? toRaster(qrow.data.data, im.raster_order, 0) : seq ? toRaster(seq, im.raster_order, im.patch_offset) : null;
  return (
    <div className="grid2">
      <div className="card">
        <h2>Vision encoder attention</h2>
        <div className="row">
          <label className="small">Layer <select value={layer} onChange={(e) => setLayer(+e.target.value)}>{layers.map((l) => <option key={l} value={l}>ViT block {l}</option>)}</select></label>
          <label className="small">Head <select value={head} onChange={(e) => setHead(+e.target.value)}><option value={-1}>mean of heads</option>{Array.from({ length: H }, (_, h) => <option key={h} value={h}>head {h}</option>)}</select></label>
          {tok.images.length > 1 && <label className="small">Image <select value={img} onChange={(e) => { setImg(+e.target.value); setQpatch(null); }}>{tok.images.map((x: any) => <option key={x.index} value={x.index}>{x.index === 0 ? "input" : `crop ${x.index}`}</option>)}</select></label>}
          {qpatch && <button onClick={() => setQpatch(null)}>clear query patch</button>}
        </div>
        <p className="small">{qpatch ? `Where patch (${qpatch[0]}, ${qpatch[1]}) looks (head-mean attention row).` : "Attention received: average attention each patch gets from all patches of the same image. Click a patch to see where it looks instead" + (hasMean && img === 0 ? "." : " (full matrix stored only for the first image).")}</p>
        <ImageHeat src={`/api/traces/${tid}/image/${img}`} gh={im.grid_thw[1]} gw={im.grid_thw[2]} values={overlay} width={Math.min(560, 20 * im.grid_thw[2])} onCell={(r, c) => setQpatch([r, c])} selected={qpatch} label="attention" />
        <Disclosure title="Tensor details">
          <pre>{`hidden [N_p=${N}, d_v]  →  qkv [N_p, 3, H=${H}, d_h]  →  scores [H, N_p, N_p] = ${H}×${N}×${N} = ${(H * N * N).toLocaleString()} floats
softmax over keys (no causal mask; images attend only within themselves via cu_seqlens)
stored: received [H, N_p] (column means), entropy [H]${hasMean ? ", head-mean matrix [N_p, N_p]" : ""}`}</pre>
        </Disclosure>
      </div>
      <div className="card">
        <h3>Per-head attention entropy (layer {layer})</h3>
        <p className="small">Low entropy = the head focuses on a few patches; high = it spreads attention broadly.</p>
        {ent.data && <Bars items={Array.from(ent.data.data).map((v, h) => ({ key: String(h), label: `head ${h}`, value: v, hint: "nats" }))} width={460} color="var(--series-3)" />}
        {vnorm.data && (<>
          <h3 style={{ marginTop: 14 }}>Residual norm across ViT depth</h3>
          <p className="small">Rows = ViT blocks, columns = patches (sequence order). Norm growth with depth is normal; outlier columns are high-norm "register-like" patches.</p>
          <Heatmap data={vnorm.data.data} rows={vnorm.data.shape[0]} cols={vnorm.data.shape[1]} width={460} height={Math.max(120, vnorm.data.shape[0] * 7)} rowLabel={(i) => `block ${i}`} colLabel={(j) => `patch ${j}`} valueLabel="‖x‖" log />
        </>)}
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- 4 merger */
function MergerStage({ tid, tok, m }: any) {
  const tidx = useTracePart<Record<string, any>>(tid, "tensors");
  const graph = useTracePart(tid, "graph");
  const merger = useTensorRowNorms(tid, tidx.data && "visual.merger" in tidx.data ? "visual.merger" : null);
  const dsKeys = Object.keys(tidx.data ?? {}).filter((k) => k.startsWith("visual.deepstack_merger_list."));
  const [ds, setDs] = useState(0);
  const dsn = useTensorRowNorms(tid, dsKeys[ds] ?? null);
  const im = tok.images[0];
  const merge = 2;
  const gn = (graph.data?.nodes ?? []).find((n: any) => n.id === "vision.merger");
  return (
    <div className="grid2">
      <div className="card">
        <h2>Patch merger: 4 patches → 1 LLM token</h2>
        <p>Each orange 2×2 window is concatenated ({gn ? gn.inputs.x[1][1] : "d_v"} × 4 = {gn ? gn.weights.fc1[0][1] : ""} numbers) and mapped by a 2-layer MLP to the LLM width. The overlay is the L2 norm of each resulting visual token.</p>
        {im && merger && <ImageHeat src={`/api/traces/${tid}/image/0`} gh={im.grid_thw[1]} gw={im.grid_thw[2]} mergeGrid={2} values={mergedToRaster(merger.slice(0, im.n_tokens), im.grid_thw[1], im.grid_thw[2], merge)} width={Math.min(560, 20 * im.grid_thw[2])} label="‖token‖" />}
        {!merger && <div className="muted small">Merger output not stored at this capture policy (stats only).</div>}
        {gn && <Disclosure title="Operation"><pre>{gn.op}</pre></Disclosure>}
      </div>
      <div className="card">
        <h2>Fusion into the text sequence</h2>
        <p>The processor inserted {m.n_vision_tokens} <code>&lt;|image_pad|&gt;</code> placeholders. After embedding, those rows are overwritten with the merger outputs, so from layer 0 on, text and image tokens are processed by the same decoder layers.</p>
        {dsKeys.length > 0 ? (<>
          <h3>DeepStack (Qwen3-VL)</h3>
          <p className="small">Features from ViT blocks {m.model.family === "qwen3_vl" ? "(e.g. 8, 16, 24)" : ""} pass through their own mergers and are <b>added</b> to the hidden states of LLM layers 0, 1, 2 at the image positions — more visual detail, no extra tokens.</p>
          <div className="row">{dsKeys.map((k, i) => <button key={k} className={ds === i ? "active" : ""} onClick={() => setDs(i)}>DeepStack {i} → LLM layer {i}</button>)}</div>
          {im && dsn && <ImageHeat src={`/api/traces/${tid}/image/0`} gh={im.grid_thw[1]} gw={im.grid_thw[2]} values={mergedToRaster(dsn.slice(0, im.n_tokens), im.grid_thw[1], im.grid_thw[2], merge)} width={Math.min(520, 18 * im.grid_thw[2])} label="‖added vector‖" />}
        </>) : <div className="warn small">{m.model.family === "qwen3_vl" ? "DeepStack tensors not stored at this policy." : "No DeepStack in this model: Qwen3.5/3.8 configs have deepstack_visual_indexes = [] — vision enters only through the merged tokens."}</div>}
      </div>
    </div>
  );
}

function useTensorRowNorms(tid: string, name: string | null): number[] | null {
  const [v, setV] = useState<number[] | null>(null);
  useEffect(() => {
    setV(null);
    if (!name) return;
    fetch(`/api/traces/${tid}/tensor/${encodeURIComponent(name)}`).then((r) => r.json()).then((j) => setV(j.row_norms)).catch(() => setV(null));
  }, [tid, name]);
  return v;
}

/* ---------------------------------------------------------------- 5 layers */
function LayerStage({ tid, tok, pidx, m }: any) {
  const [metric, setMetric] = useState<"residual_norm" | "residual_rel_update" | "residual_cos_prev">("residual_rel_update");
  const heat = usePreview(tid, metric);
  const graph = useTracePart(tid, "graph");
  const layerTypes: string[] = useMemo(() => (graph.data?.nodes ?? []).filter((n: any) => n.level === "block" && n.id.startsWith("llm.")).map((n: any) => n.meta.mixer), [graph.data]);
  const [layer, setLayer] = useState(0);
  const lt = layerTypes[layer];
  const T = tok.ids.length;
  const firstVis = tok.segments.indexOf("vision");
  const lastVis = tok.segments.lastIndexOf("vision");
  return (
    <div>
      <div className="card">
        <h2>Residual stream through the decoder</h2>
        <div className="row">
          {(["residual_rel_update", "residual_norm", "residual_cos_prev"] as const).map((k) => (
            <button key={k} className={metric === k ? "active" : ""} onClick={() => setMetric(k)}>{{ residual_norm: "‖h‖ per token", residual_rel_update: "relative update ‖Δh‖/‖h‖", residual_cos_prev: "cos(h_l, h_{l−1})" }[k]}</button>))}
        </div>
        <p className="small">Rows = layers (click to inspect), columns = tokens. Vision tokens span columns {firstVis}–{lastVis}. Layer types: <span className="pill tag-gdn">GDN</span> <span className="pill tag-attn">attention</span>.</p>
        {heat.data && <Heatmap data={heat.data.data} rows={heat.data.shape[0]} cols={heat.data.shape[1]} width={1100} height={Math.max(160, heat.data.shape[0] * 9)}
          rowLabel={(i) => `layer ${i} (${layerTypes[i] === "linear_attention" ? "GDN" : "attn"})`} colLabel={(j) => `tok ${j} ${JSON.stringify(tok.strings[j])} [${tok.segments[j]}]`} valueLabel={metric} onCell={(i) => setLayer(i)} log={metric !== "residual_cos_prev"} />}
      </div>
      <div className="card">
        <div className="row"><h2>Layer {layer}</h2><span className={`pill ${lt === "linear_attention" ? "tag-gdn" : "tag-attn"}`}>{lt === "linear_attention" ? "Gated DeltaNet" : "full attention"}</span>
          <input type="range" min={0} max={layerTypes.length - 1} value={layer} onChange={(e) => setLayer(+e.target.value)} style={{ width: 300 }} /></div>
        {lt === "linear_attention" ? <GDNPanel tid={tid} layer={layer} tok={tok} /> : <AttnPanel tid={tid} layer={layer} tok={tok} pidx={pidx} />}
        {pidx[`moe.${layer}.experts`] && <MoEPanel tid={tid} layer={layer} tok={tok} />}
      </div>
    </div>
  );
}

function AttnPanel({ tid, layer, tok, pidx }: any) {
  const key = `attn.${layer}`;
  const has = !!pidx[`${key}.sink`];
  const sink = usePreview(has ? tid : null, `${key}.sink`);
  const last = usePreview(has ? tid : null, `${key}.last_row`);
  const toImg = usePreview(pidx[`${key}.answer_to_image`] ? tid : null, `${key}.answer_to_image`);
  const gate = usePreview(pidx[`attn_gate_mean.${layer}`] ? tid : null, `attn_gate_mean.${layer}`);
  const T = tok.ids.length;
  const stride = T > 600 ? 2 : 1;
  const mean = usePreview(pidx[`${key}.mean`] ? tid : null, `${key}.mean`, { stride });
  const [head, setHead] = useState(-1);
  if (!has) return <div className="muted">Attention not recorded for this layer at this policy (lite policy records every 4th layer).</div>;
  const H = sink.data?.shape[0] ?? 0;
  const im = tok.images[0];
  let imgVals: number[] | null = null;
  if (toImg.data && im) {
    const n = toImg.data.shape[1];
    const row = head < 0 ? Array.from({ length: n }, (_, j) => { let s = 0; for (let h = 0; h < H; h++) s += toImg.data!.data[h * n + j]; return s / H; }) : rowOf(toImg.data, head);
    imgVals = mergedToRaster(row.slice(0, im.n_tokens), im.grid_thw[1], im.grid_thw[2], 2);
  }
  return (
    <div className="grid2">
      <div>
        <h3>Where do answer tokens look in the image?</h3>
        <label className="small">Head <select value={head} onChange={(e) => setHead(+e.target.value)}><option value={-1}>mean</option>{Array.from({ length: H }, (_, h) => <option key={h} value={h}>head {h}</option>)}</select></label>
        {imgVals && im ? <ImageHeat src={`/api/traces/${tid}/image/0`} gh={im.grid_thw[1]} gw={im.grid_thw[2]} values={imgVals} width={Math.min(520, 18 * im.grid_thw[2])} label="attention mass" />
          : <div className="muted small">No answer tokens / image in this run.</div>}
        <p className="small">Attention probability from every generated-answer position to each image token, averaged over answer positions, projected back onto the 2×2 merge windows.</p>
        {gate.data && (<>
          <h3>Gated-attention gate σ(XW) (mean over head dims)</h3>
          <Heatmap data={gate.data.data} rows={gate.data.shape[0]} cols={gate.data.shape[1]} width={520} height={160} rowLabel={(i) => `tok ${i} ${JSON.stringify(tok.strings[i])}`} colLabel={(j) => `head ${j}`} valueLabel="gate" domain={[0, 1]} />
          <p className="small">Rows = tokens, columns = heads. Values near 0 switch a head off for that token — this is what lets heads avoid the attention sink.</p>
        </>)}
      </div>
      <div>
        <h3>Attention sink: mean attention to token 0, per head</h3>
        {sink.data && <Bars items={Array.from(sink.data.data).map((v, h) => ({ key: String(h), label: `head ${h}`, value: v }))} width={460} color="var(--series-2)" fmt={(v) => (v * 100).toFixed(1) + "%"} />}
        <h3 style={{ marginTop: 12 }}>Last prompt token → all keys</h3>
        {last.data && <Heatmap data={last.data.data} rows={last.data.shape[0]} cols={last.data.shape[1]} width={520} height={Math.max(60, H * 8)} rowLabel={(i) => `head ${i}`} colLabel={(j) => `tok ${j} ${JSON.stringify(tok.strings[j])} [${tok.segments[j]}]`} valueLabel="p" log />}
        {mean.data && (<Disclosure title={`Head-mean attention matrix [${T}×${T}]${stride > 1 ? ` (every ${stride}th row/col)` : ""}`}>
          <Heatmap data={mean.data.data} rows={mean.data.shape[0]} cols={mean.data.shape[1]} width={520} height={520} rowLabel={(i) => `query ${i * stride}`} colLabel={(j) => `key ${j * stride}`} valueLabel="p" log />
        </Disclosure>)}
        <Disclosure title="Tensor details">
          <pre>{`Q [T=${T}, H_q, d_h] · Kᵀ → scores [H_q, ${T}, ${T}] (causal mask)
stored per layer: sink [H_q], entropy [H_q], last_row [H_q, T], answer_to_image [H_q, N_v]${mean.data ? `, mean [T, T]` : ""}`}</pre>
        </Disclosure>
      </div>
    </div>
  );
}

function GDNPanel({ tid, layer, tok }: any) {
  const [what, setWhat] = useState<"alpha" | "beta" | "state_norm" | "write_norm">("state_norm");
  const a = usePreview(tid, `gdn.${layer}.${what}`);
  const sn = usePreview(tid, `gdn.${layer}.state_norm`);
  const T = tok.ids.length;
  const meanOver = (x: Arr | undefined) => x ? Array.from({ length: x.shape[0] }, (_, t) => { let s = 0; for (let h = 0; h < x.shape[1]; h++) s += x.data[t * x.shape[1] + h]; return s / x.shape[1]; }) : [];
  const firstVis = tok.segments.indexOf("vision");
  const lastVis = tok.segments.lastIndexOf("vision");
  const desc = { alpha: "decay α_t ∈ (0,1): how much of the memory survives this token (1 = keep everything)", beta: "write strength β_t ∈ (0,1): how strongly this token writes its correction", state_norm: "‖S_t‖_F: size of the memory matrix after this token", write_norm: "‖β(v − Sᵀk)‖: size of the correction written (the delta rule's error)" }[what];
  return (
    <div className="grid2">
      <div>
        <div className="row">{(["state_norm", "write_norm", "alpha", "beta"] as const).map((k) => <button key={k} className={what === k ? "active" : ""} onClick={() => setWhat(k)}>{k}</button>)}</div>
        <p className="small">{desc}. Rows = tokens, columns = value heads.</p>
        {a.data && <Heatmap data={a.data.data} rows={a.data.shape[0]} cols={a.data.shape[1]} width={520} height={300} rowLabel={(i) => `tok ${i} ${JSON.stringify(tok.strings[i])} [${tok.segments[i]}]`} colLabel={(j) => `head ${j}`} valueLabel={what} />}
      </div>
      <div>
        <h3>Memory size along the sequence (mean over heads)</h3>
        <LineChart width={520} height={220} xLabel="token" yLabel="‖S‖" series={[{ name: "state norm", values: meanOver(sn.data), color: "var(--series-1)" }]}
          marks={firstVis >= 0 ? [{ x: firstVis, label: "image" }, { x: lastVis, label: "end image" }] : []} />
        <TeX block tex={String.raw`S_t=\alpha_t S_{t-1}+\beta_t\,k_t\big(v_t-S_{t-1}^{\top}k_t\big)^{\top},\qquad y_t=S_t^{\top}q_t`} />
        <Disclosure title="Tensor details">
          <pre>{`per head: S ∈ R^{d_k×d_v} (128×128), q,k ∈ R^128 (L2-normalised), α,β scalars
inputs: qkv [T=${T}, 2·H_k·d_k + H_v·d_v] → short conv (k=4) → SiLU → split
state is O(1) in sequence length — this is why 3/4 of layers need no KV cache`}</pre>
        </Disclosure>
      </div>
    </div>
  );
}

function MoEPanel({ tid, layer, tok }: any) {
  const ex = usePreview(tid, `moe.${layer}.experts`);
  const w = usePreview(tid, `moe.${layer}.weights`);
  const load = usePreview(tid, `moe.${layer}.load`);
  if (!ex.data || !w.data || !load.data) return null;
  const [T, k] = ex.data.shape; const E = load.data.shape[0];
  const grid = new Float32Array(T * E);
  for (let t = 0; t < T; t++) for (let j = 0; j < k; j++) grid[t * E + ex.data.data[t * k + j]] = w.data.data[t * k + j];
  const top = Array.from(load.data.data).map((v, e) => ({ e, v })).sort((a, b) => b.v - a.v).slice(0, 15);
  return (
    <div className="grid2" style={{ marginTop: 12 }}>
      <div><h3>Routing: token × expert (weight of chosen experts)</h3>
        <Heatmap data={grid} rows={T} cols={E} width={520} height={300} rowLabel={(i) => `tok ${i} ${JSON.stringify(tok.strings[i])} [${tok.segments[i]}]`} colLabel={(j) => `expert ${j}`} valueLabel="routing weight" /></div>
      <div><h3>Most-used experts in this layer</h3>
        <Bars items={top.map(({ e, v }) => ({ key: String(e), label: `expert ${e}`, value: v, hint: "tokens routed" }))} color="var(--series-7)" width={460} fmt={(v) => String(v)} /></div>
    </div>
  );
}

/* ---------------------------------------------------------------- 6 lens */
function LensStage({ tid, tok, out }: any) {
  const tp = usePreview(tid, "logit_lens_target_p");
  const ids = usePreview(tid, "logit_lens_ids");
  const probs = usePreview(tid, "logit_lens_probs");
  const rows: number[] = tok.lens_rows;
  const ansCols = rows.map((r, j) => [r, j] as const).filter(([r]) => r + 1 < tok.ids.length && ["answer", "thinking"].includes(tok.segments[r + 1])).map(([, j]) => j);
  const [col, setCol] = useState<number | null>(null);
  const c = col ?? ansCols[0] ?? rows.length - 1;
  if (!tp.data || !ids.data || !probs.data) return <Loading what="logit lens" />;
  const L = tp.data.shape[0], R = tp.data.shape[1];
  const sub = ansCols.length ? ansCols : Array.from({ length: Math.min(R, 64) }, (_, j) => R - Math.min(R, 64) + j);
  const data = new Float32Array(L * sub.length);
  for (let l = 0; l < L; l++) sub.forEach((j, q) => { data[l * sub.length + q] = tp.data!.data[l * R + j]; });
  const nextTok = (j: number) => tok.strings[rows[j] + 1] ?? "";
  return (
    <div className="grid2">
      <div className="card">
        <h2>When does the model "know" the next token?</h2>
        <p className="small">Each intermediate residual stream is decoded with the final norm + LM head. Cell = probability of the token that actually came next. Columns = answer positions (click one).</p>
        <Heatmap data={data} rows={L} cols={sub.length} width={560} height={Math.max(160, L * 9)} rowLabel={(i) => `layer ${i}`} colLabel={(q) => `→ ${JSON.stringify(nextTok(sub[q]))}`} valueLabel="p(next)" domain={[0, 1]} onCell={(_, q) => setCol(sub[q])} highlightCol={sub.indexOf(c)} />
      </div>
      <div className="card">
        <h3>Top-5 per layer at position {rows[c]} (next token {JSON.stringify(nextTok(c))})</h3>
        <table><thead><tr><th>layer</th>{[1, 2, 3, 4, 5].map((k) => <th key={k}>#{k}</th>)}</tr></thead>
          <tbody>{Array.from({ length: L }, (_, l) => (
            <tr key={l}><td>{l}</td>{[0, 1, 2, 3, 4].map((k) => { const id = ids.data!.data[(l * R + c) * 5 + k]; const p = probs.data!.data[(l * R + c) * 5 + k];
              return <td key={k} style={{ background: `color-mix(in srgb, var(--series-1) ${Math.round(p * 60)}%, transparent)` }}>{showTok(out.lens_vocab?.[id] ?? String(id))} <span className="muted small">{(p * 100).toFixed(0)}%</span></td>; })}</tr>))}</tbody></table>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- 7 output */
function OutputStage({ tid, tok, out, m }: any) {
  const lp = usePreview(tid, "token_logprob");
  const ent = usePreview(tid, "token_entropy");
  const steps = out.steps.flat();
  const { tip, show, hide } = useTip();
  const ans = tok.segments.map((s: string, i: number) => [s, i] as const).filter(([s]: any) => s === "answer" || s === "thinking").map(([, i]: any) => i);
  return (
    <div>
      <div className="card">
        <div className="row"><h2>Generated reply</h2>
          {out.rule_score === 1 && <span className="pill" style={{ color: "var(--good)", borderColor: "var(--good)" }}>✓ exact match</span>}
          {out.rule_score === 0 && <span className="pill" style={{ color: "var(--critical)", borderColor: "var(--critical)" }}>✗ wrong letter</span>}
          {out.rule_score == null && <span className="pill">needs judge ({out.rule_source})</span>}
          <span className="small muted">extracted: <b>{String(out.answer_extracted)}</b> · gold: {out.gold.slice(0, 3).join(" / ")} · {out.rounds} round(s){out.boxes.length ? ` · crops ${JSON.stringify(out.boxes)}` : ""}</span>
        </div>
        <p className="small">Greedy decoding. Bar under each token = its probability; hover for the top-10 alternatives the model considered.</p>
        <div className="tokstrip">
          {steps.map((s: any, i: number) => (
            <span key={i} className="tok answer" style={{ borderBottom: `3px solid color-mix(in srgb, var(--series-1) ${Math.round(Math.exp(s.logprob) * 100)}%, var(--critical))` }}
              onMouseMove={(e) => show(e, <table><tbody>{s.top.map(([id, t, l]: any) => <tr key={id}><td>{showTok(t)}</td><td className="num">{(Math.exp(l) * 100).toFixed(1)}%</td></tr>)}</tbody></table>)} onMouseLeave={hide}>{showTok(s.token)}</span>))}
        </div>
        {tip}
      </div>
      {lp.data && ent.data && ans.length > 0 && (
        <div className="card">
          <h3>Teacher-forced log-probability and entropy over the answer</h3>
          <div className="grid2">
            <LineChart width={560} height={200} xLabel="answer token" yLabel="log p" series={[{ name: "log p(token)", values: ans.map((i: number) => lp.data!.data[i]), color: "var(--series-1)" }]} xTick={(i) => showTok(tok.strings[ans[i]] ?? "")} />
            <LineChart width={560} height={200} xLabel="answer token" yLabel="entropy (nats)" series={[{ name: "entropy", values: ans.map((i: number) => ent.data!.data[i - 1]), color: "var(--series-2)" }]} xTick={(i) => showTok(tok.strings[ans[i]] ?? "")} />
          </div>
        </div>)}
      <div className="card"><h3>Raw reply</h3><pre style={{ whiteSpace: "pre-wrap", maxHeight: 320, overflow: "auto" }}>{out.reply.replace(/\n/g, "↵\n").slice(0, 6000)}</pre>
        {m.mode === "r3" && <Disclosure title="Round-by-round parts (crop loop)"><pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(out.parts, null, 1)}</pre></Disclosure>}
        <div className="small muted">{fmtNum(m.T)} tokens total, generation {m.timing.generate_s?.toFixed(1)}s.</div></div>
    </div>
  );
}
