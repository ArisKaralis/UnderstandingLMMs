import { useMemo, useState } from "react";
import { PIPELINES, Stage } from "../content/pipelines";
import { Algo, ALGO_INFO, defaults, gradWeight, groupAdvantages, Params, surrogate } from "../lib/rl";
import { useTraining, useTrainingFile } from "../lib/api";
import { LineChart, Heatmap } from "../components/charts";
import { Disclosure, Empty, TeX, useTip } from "../components/ui";
import { fmtNum, showTok } from "../lib/format";

const TABS = ["Pipelines", "Algorithm playground", "RL step viewer", "SFT step viewer", "Training curves"];

export default function TrainingPage() {
  const [tab, setTab] = useState(0);
  return (
    <div>
      <div className="card">
        <h1>Training lab</h1>
        <p>How Qwen-VL models are pre-trained and post-trained (SFT, distillation, RL), with the inputs and outputs of each step. The step viewers replay real tiny-scale runs on your VLIR data (same samples and LoRA recipe as MLLM_Testing).</p>
        <div className="stagenav">{TABS.map((t, i) => <button key={t} className={tab === i ? "active" : ""} onClick={() => setTab(i)}>{t}</button>)}</div>
      </div>
      {tab === 0 && <Pipelines />}
      {tab === 1 && <Playground />}
      {tab === 2 && <RLViewer />}
      {tab === 3 && <SFTViewer />}
      {tab === 4 && <Curves />}
    </div>
  );
}

function Pipelines() {
  const [fam, setFam] = useState<keyof typeof PIPELINES>("qwen3_vl");
  const p = PIPELINES[fam];
  const [sel, setSel] = useState<Stage>(p.stages[0]);
  return (
    <div className="split">
      <div className="card">
        <div className="row">{Object.entries(PIPELINES).map(([k, v]) => <button key={k} className={fam === k ? "active" : ""} onClick={() => { setFam(k as any); setSel(PIPELINES[k].stages[0]); }}>{v.label}</button>)}</div>
        {p.notes.map((n) => <p key={n} className="small">{n}</p>)}
        {(["pre", "post"] as const).map((ph) => (
          <div key={ph}>
            <h3>{ph === "pre" ? "Pre-training" : "Post-training"}</h3>
            <div className="macro">{p.stages.filter((s) => s.phase === ph).map((s, i, arr) => (
              <div key={s.id}>
                <button className={`node ${sel.id === s.id ? "selected" : ""}`} onClick={() => setSel(s)}>
                  <div className="row" style={{ justifyContent: "space-between" }}><b>{s.name}</b>
                    <span className={`pill ${s.status === "published" ? "tag-vision" : s.status === "vendor-claim" ? "" : "tag-moe"}`}>{s.status}</span></div>
                  <div className="meta">{s.trains}{s.tokens ? ` · ${s.tokens} tokens` : ""}{s.seq ? ` · seq ${s.seq}` : ""}</div>
                </button>{i < arr.length - 1 && <div className="flowarrow">↓</div>}
              </div>))}</div>
          </div>))}
      </div>
      <div className="card sticky">
        <h2>{sel.name}</h2>
        <table><tbody>
          <tr><td>What is trained</td><td style={{ whiteSpace: "normal" }}>{sel.trains}</td></tr>
          <tr><td>Data</td><td style={{ whiteSpace: "normal" }}>{sel.data}</td></tr>
          {sel.tokens && <tr><td>Tokens</td><td>{sel.tokens}</td></tr>}
          {sel.seq && <tr><td>Sequence length</td><td>{sel.seq}</td></tr>}
          <tr><td>Objective</td><td style={{ whiteSpace: "normal" }}>{sel.objective}</td></tr>
        </tbody></table>
        {sel.eq && <div style={{ overflowX: "auto", padding: "8px 0" }}><TeX block tex={sel.eq} /></div>}
        <Disclosure title="Inputs → outputs" open><p>{sel.io}</p></Disclosure>
        <p className="small muted">Source: {sel.source}</p>
        {sel.status === "vendor-claim" && <div className="warn small">Not described in a technical report — quoted from the model card; the exact method is not disclosed.</div>}
      </div>
    </div>
  );
}

function Playground() {
  const [p, setP] = useState<Params>(defaults);
  const [sign, setSign] = useState<1 | -1>(1);
  const [show, setShow] = useState<Algo[]>(["grpo", "dapo", "gspo", "sapo"]);
  const [rewards, setRewards] = useState("1, 0, 0, 1, 0, 0, 0, 1");
  const rs = useMemo(() => rewards.split(/[,\s]+/).filter(Boolean).map(Number).filter(Number.isFinite), [rewards]);
  const adv = groupAdvantages(rs.length ? rs : [0]);
  const xs = Array.from({ length: 121 }, (_, i) => 0.4 + i * 0.01);
  const colors: Record<Algo, string> = { grpo: "var(--series-1)", dapo: "var(--series-2)", gspo: "var(--series-3)", sapo: "var(--series-7)" };
  const slider = (k: keyof Params, min: number, max: number, step: number) => (
    <label className="small row" key={k}>{k} <input type="range" min={min} max={max} step={step} value={p[k]} onChange={(e) => setP({ ...p, [k]: +e.target.value })} /> {p[k]}</label>);
  return (
    <div>
      <div className="card">
        <h2>Group advantages</h2>
        <p className="small">Rewards of the G responses sampled for one prompt (edit). Each response's advantage is its reward standardised within the group; every token of that response gets the same advantage. All-equal rewards give zero advantage — DAPO's dynamic sampling drops such groups.</p>
        <input type="text" value={rewards} onChange={(e) => setRewards(e.target.value)} style={{ width: 360 }} />
        <table style={{ marginTop: 8, width: "auto" }}><thead><tr><th>response</th>{rs.map((_, i) => <th key={i} className="num">{i}</th>)}</tr></thead>
          <tbody><tr><td>reward</td>{rs.map((r, i) => <td key={i} className="num">{r}</td>)}</tr>
            <tr><td>advantage Â</td>{adv.map((a, i) => <td key={i} className="num" style={{ color: a > 0 ? "var(--series-1)" : a < 0 ? "var(--series-8)" : undefined }}>{a.toFixed(2)}</td>)}</tr></tbody></table>
        <TeX block tex={String.raw`\hat A_i=\frac{r_i-\mathrm{mean}(r_{1..G})}{\mathrm{std}(r_{1..G})}`} />
      </div>
      <div className="card">
        <h2>Objective and gradient weight vs. importance ratio r = π_θ / π_old</h2>
        <div className="row">
          <button className={sign > 0 ? "active" : ""} onClick={() => setSign(1)}>Â &gt; 0 (good response)</button>
          <button className={sign < 0 ? "active" : ""} onClick={() => setSign(-1)}>Â &lt; 0 (bad response)</button>
          {(Object.keys(ALGO_INFO) as Algo[]).map((a) => (<label key={a} className="small"><input type="checkbox" checked={show.includes(a)} onChange={(e) => setShow(e.target.checked ? [...show, a] : show.filter((x) => x !== a))} /> {ALGO_INFO[a].name}</label>))}
        </div>
        <div className="row">{slider("epsLow", 0.05, 0.4, 0.01)}{slider("epsHigh", 0.05, 0.5, 0.01)}{slider("tauPos", 0.2, 4, 0.05)}{slider("tauNeg", 0.2, 4, 0.05)}</div>
        <div className="grid2">
          <LineChart width={560} height={260} xLabel="ratio r" yLabel="surrogate f(r)·Â" x0={0} xTick={(i) => xs[i]?.toFixed(2) ?? ""}
            series={show.map((a) => ({ name: ALGO_INFO[a].name, values: xs.map((r) => surrogate(a, r, sign, p)), color: colors[a] }))} />
          <LineChart width={560} height={260} xLabel="ratio r" yLabel="gradient weight" x0={0} xTick={(i) => xs[i]?.toFixed(2) ?? ""} yDomain={[0, 1.05]}
            series={show.map((a) => ({ name: ALGO_INFO[a].name, values: xs.map((r) => gradWeight(a, r, sign, p)), color: colors[a] }))} />
        </div>
        <p className="small">Hard clipping (GRPO/DAPO/GSPO) makes the gradient exactly 0 once r leaves the trust region on the side the advantage pushes toward; SAPO's sigmoid gate decays smoothly instead. GSPO's band is so narrow (3e-4/4e-4) because its ratio is a per-token geometric mean over the whole response.</p>
      </div>
      <div className="grid2">{(Object.keys(ALGO_INFO) as Algo[]).map((a) => (
        <div key={a} className="card"><h3>{ALGO_INFO[a].name} <span className="small muted">{ALGO_INFO[a].ref}</span></h3><p className="small">{ALGO_INFO[a].text}</p>
          <div style={{ overflowX: "auto" }}><TeX block tex={ALGO_INFO[a].eq} /></div></div>))}
      </div>
    </div>
  );
}

function useRun(kind: string) {
  const runs = useTraining();
  const list = (runs.data ?? []).filter((r) => r.kind === kind);
  const [rid, setRid] = useState<string | null>(null);
  const run = list.find((r) => r.id === rid) ?? list[0];
  const picker = list.length ? (
    <label className="small">Run <select value={run?.id} onChange={(e) => setRid(e.target.value)}>{list.map((r) => <option key={r.id} value={r.id}>{r.id} · {r.model} · {r.algo ?? kind}{r.demo ? " (demo)" : ""}</option>)}</select></label>) : null;
  return { run, picker, loading: runs.isLoading };
}

function RLViewer() {
  const { run, picker, loading } = useRun("rl");
  const steps = useTrainingFile<string[]>(run?.id ?? null, "steps");
  const [step, setStep] = useState<string | null>(null);
  const s = step ?? steps.data?.[0] ?? null;
  const data = useTrainingFile<any>(run?.id ?? null, s ? `steps/${s}` : null);
  const [algo, setAlgo] = useState<Algo>("grpo");
  const [resp, setResp] = useState(0);
  const { tip, show, hide } = useTip();
  if (loading) return null;
  if (!run) return <Empty>No RL runs yet. Run <code>python -m lmmviz.train_lab.rl</code> (laptop smoke run or <code>scripts/hpc/train_rl.slurm</code>) and copy <code>data/training/&lt;run&gt;</code> here.</Empty>;
  const d = data.data;
  const rc = d?.recomputed?.[algo];
  const g = d?.group?.[resp];
  return (
    <div>
      <div className="card">
        <div className="row">{picker}
          <label className="small">Step <select value={s ?? ""} onChange={(e) => setStep(e.target.value)}>{(steps.data ?? []).map((x) => <option key={x} value={x}>{x.replace(".json", "")}</option>)}</select></label>
          <span className="small muted">trained with <b>{run.algo}</b>; view the same step as →</span>
          {(Object.keys(ALGO_INFO) as Algo[]).map((a) => <button key={a} className={algo === a ? "active" : ""} onClick={() => setAlgo(a)}>{ALGO_INFO[a].name}</button>)}
        </div>
        {run.demo && <div className="warn small">Demo run (tiny model / few steps): numbers show the mechanics, not a useful policy.</div>}
      </div>
      {d && (<>
        <div className="grid2">
          <div className="card">
            <h3>1 · Prompt</h3>
            <div className="row">{d.image && <img src={`/api/training/${run.id}/${d.image}`} style={{ maxWidth: 220, maxHeight: 220, borderRadius: 4 }} />}
              <pre style={{ whiteSpace: "pre-wrap", flex: 1, maxHeight: 220 }}>{d.question}</pre></div>
            <div className="small muted">gold: {JSON.stringify(d.gold)} {d.gold_bbox ? `· gold box ${JSON.stringify(d.gold_bbox)}` : ""}</div>
          </div>
          <div className="card">
            <h3>2 · {d.group.length} sampled responses → rewards → advantages</h3>
            <table><thead><tr><th>#</th><th>reward</th><th>parts</th><th>Â</th><th>tokens</th>{rc?.seq_ratio && <th>s_i</th>}</tr></thead>
              <tbody>{d.group.map((x: any, i: number) => (
                <tr key={i} onClick={() => setResp(i)} style={{ cursor: "pointer", background: i === resp ? "var(--surface-2)" : undefined }}>
                  <td>{i}</td><td className="num">{x.reward.toFixed(2)}</td><td className="small">{Object.entries(x.reward_parts).map(([k, v]: any) => `${k} ${(+v).toFixed(2)}`).join(" · ")}</td>
                  <td className="num" style={{ color: rc?.advantages[i] > 0 ? "var(--series-1)" : "var(--series-8)" }}>{rc ? rc.advantages[i].toFixed(2) : ""}</td><td className="num">{x.tokens.length}</td>
                  {rc?.seq_ratio && <td className="num">{rc.seq_ratio[i].toFixed(5)}</td>}</tr>))}</tbody></table>
            {rc && <div className="kpi" style={{ marginTop: 8 }}><div><b>{fmtNum(rc.loss, 4)}</b><span>{ALGO_INFO[algo].name} loss</span></div><div><b>{(rc.clip_fraction * 100).toFixed(1)}%</b><span>{algo === "sapo" ? "tokens with gate weight < ½" : "clipped tokens"}</span></div></div>}
          </div>
        </div>
        {g && rc && (
          <div className="card">
            <h3>3 · Response {resp}, token by token under {ALGO_INFO[algo].name}</h3>
            <p className="small">Colour = per-token objective contribution (blue pushes probability up, red down); a struck-through token is clipped (zero gradient) {algo === "sapo" ? "/ softly gated" : ""}. Hover for ratio, log-probs, KL.</p>
            <div className="tokstrip">{g.tokens.map((t: string, j: number) => {
              const v = rc.per_token_objective[resp][j]; const clipped = rc.clipped[resp][j] > 0;
              return (<span key={j} className="tok" style={{ background: `color-mix(in srgb, ${v >= 0 ? "var(--series-1)" : "var(--series-8)"} ${Math.min(70, Math.abs(v) * 60)}%, transparent)`, textDecoration: clipped ? "line-through" : undefined }}
                onMouseMove={(e) => show(e, <div><b>{JSON.stringify(t)}</b><br />ratio r = {rc.ratio[resp][j].toFixed(4)}<br />log π = {g.logp[j].toFixed(3)} · log π_old = {g.logp_old[j].toFixed(3)}{g.logp_ref ? <> · log π_ref = {g.logp_ref[j].toFixed(3)}</> : null}<br />objective {v.toFixed(4)}{rc.kl ? <><br />KL {rc.kl[resp][j].toFixed(4)}</> : null}{rc.gate_weight ? <><br />gate weight {rc.gate_weight[resp][j].toFixed(3)}</> : null}</div>)} onMouseLeave={hide}>{showTok(t)}</span>);
            })}</div>{tip}
            <Disclosure title="Tensor details for this step">
              <pre>{`G=${d.group.length} responses, padded length L=${Math.max(...d.group.map((x: any) => x.tokens.length))}
logp, logp_old${d.group[0].logp_ref ? ", logp_ref" : ""}, mask : [G, L]     rewards, advantages : [G]
ratio r = exp(logp − logp_old) : [G, L]   ${algo === "gspo" ? "s_i = exp(mean_t log r) : [G]" : ""}
loss = −mean_i (1/|y_i|) Σ_t ${algo === "sapo" ? "f(r)·Â" : "min(r·Â, clip(r)·Â)"}${algo === "dapo" ? "   (DAPO: Σ over all tokens / total tokens)" : ""}`}</pre>
            </Disclosure>
          </div>)}
      </>)}
    </div>
  );
}

function SFTViewer() {
  const { run, picker, loading } = useRun("sft");
  const steps = useTrainingFile<string[]>(run?.id ?? null, "steps");
  const [step, setStep] = useState<string | null>(null);
  const s = step ?? steps.data?.[0] ?? null;
  const d = useTrainingFile<any>(run?.id ?? null, s ? `steps/${s}` : null).data;
  const { tip, show, hide } = useTip();
  if (loading) return null;
  if (!run) return <Empty>No SFT runs yet. Run <code>python -m lmmviz.train_lab.sft</code> and copy <code>data/training/&lt;run&gt;</code> here.</Empty>;
  const ex = d?.examples?.[0];
  const maxLoss = ex ? Math.max(...ex.token_loss.filter((x: number | null) => x != null)) : 1;
  return (
    <div>
      <div className="card"><div className="row">{picker}<label className="small">Step <select value={s ?? ""} onChange={(e) => setStep(e.target.value)}>{(steps.data ?? []).map((x) => <option key={x} value={x}>{x.replace(".json", "")}</option>)}</select></label></div>
        {run.demo && <div className="warn small">Demo run (tiny settings): mechanics only.</div>}</div>
      {ex && (<div className="grid2">
        <div className="card">
          <h3>Input → labels: which tokens get loss</h3>
          <p className="small">Grey = masked (label −100: system, user prompt, image tokens). Coloured = assistant tokens; darker = higher per-token loss −log p(target). Hover for values.</p>
          <div className="tokstrip">{ex.tokens.map((t: string, j: number) => {
            const masked = !ex.label_mask[j]; const l = ex.token_loss[j];
            if (ex.segments?.[j] === "vision" && j > 0 && ex.segments[j - 1] === "vision") return null;
            if (ex.segments?.[j] === "vision") { const n = ex.segments.slice(j).findIndex((x: string) => x !== "vision"); return <span key={j} className="tok vision">▦ ×{n < 0 ? ex.segments.length - j : n}</span>; }
            return (<span key={j} className={`tok ${masked ? "special" : ""}`} style={masked ? {} : { background: `color-mix(in srgb, var(--series-2) ${Math.round(15 + 70 * (l ?? 0) / (maxLoss || 1))}%, transparent)` }}
              onMouseMove={(e) => show(e, <span>{JSON.stringify(t)}<br />{masked ? "masked (no loss)" : `loss ${l?.toFixed(3)} · p = ${Math.exp(-(l ?? 0)).toFixed(3)}`}</span>)} onMouseLeave={hide}>{showTok(t)}</span>);
          })}</div>{tip}
          <TeX block tex={String.raw`\mathcal L=-\frac{1}{|\mathcal A|}\sum_{t\in\mathcal A}\log\pi_\theta(y_t\mid y_{<t},\text{image})`} />
        </div>
        <div className="card">
          <h3>Gradient norm per module (this step)</h3>
          {d.grad_norms && <Heatmap data={Object.values(d.grad_norms) as number[]} rows={1} cols={Object.keys(d.grad_norms).length} width={520} height={40} colLabel={(j) => Object.keys(d.grad_norms)[j]} valueLabel="‖∇‖" log />}
          {d.lora_delta && (<><h3>LoRA update size ‖BA‖ per layer</h3>
            <LineChart width={520} height={200} xLabel="layer" yLabel="‖ΔW‖_F" series={Object.entries(d.lora_delta as Record<string, number[]>).map(([k, v], i) => ({ name: k, values: v, color: `var(--series-${(i % 8) + 1})` }))} /></>)}
          <div className="small muted">loss {fmtNum(d.loss, 4)} · lr {d.lr}</div>
        </div>
      </div>)}
    </div>
  );
}

function Curves() {
  const runs = useTraining();
  const [rid, setRid] = useState<string | null>(null);
  const run = (runs.data ?? []).find((r) => r.id === rid) ?? runs.data?.[0];
  const metrics = useTrainingFile<any[]>(run?.id ?? null, "metrics.jsonl");
  if (!run) return <Empty>No training runs yet.</Empty>;
  const ms = metrics.data ?? [];
  const keys = Array.from(new Set(ms.flatMap((m) => Object.keys(m)))).filter((k) => k !== "step" && typeof ms[0]?.[k] === "number");
  return (
    <div className="card">
      <label className="small">Run <select value={run.id} onChange={(e) => setRid(e.target.value)}>{(runs.data ?? []).map((r) => <option key={r.id} value={r.id}>{r.id}</option>)}</select></label>
      <div className="grid2">{keys.map((k) => (
        <div key={k}><h4>{k}</h4><LineChart width={540} height={170} xLabel="step" series={[{ name: k, values: ms.map((m) => m[k]), color: "var(--series-1)" }]} xTick={(i) => String(ms[i]?.step ?? i)} /></div>))}</div>
    </div>
  );
}
