import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { get, useInterventions, useTraces } from "../lib/api";
import { LineChart } from "../components/charts";
import { Disclosure, Empty, Loading } from "../components/ui";
import { useApp } from "../lib/store";
import { showTok } from "../lib/format";

export default function ExperimentsPage() {
  const traces = useTraces();
  const ivs = useInterventions();
  const model = useApp((s) => s.model);
  const groups = useMemo(() => {
    const g: Record<string, typeof rows> = {};
    const rows = traces.data ?? [];
    rows.forEach((t) => { (g[`${t.demo ? "demo · " : ""}${t.model}__${t.sample}__${t.mode}`] ||= []).push(t); });
    return g;
  }, [traces.data]);
  const keys = Object.keys(groups).sort((a, b) => (+b.startsWith(model) - +a.startsWith(model)) || (+a.startsWith("demo") - +b.startsWith("demo")));
  const [gk, setGk] = useState<string | null>(null);
  const k = gk ?? keys.find((x) => groups[x].length > 1) ?? keys[0];
  const runs = k ? groups[k] : [];
  const base = runs.find((r) => r.intervention === "baseline") ?? runs[0];
  const [vid, setVid] = useState<string | null>(null);
  const variant = runs.find((r) => r.id === vid) ?? runs.find((r) => r.id !== base?.id);
  const diff = useQuery({ queryKey: ["diff", base?.id, variant?.id], queryFn: () => get<any>(`/api/diff?a=${base!.id}&b=${variant!.id}`), enabled: !!base && !!variant && base.id !== variant.id });
  if (traces.isLoading) return <Loading what="runs" />;
  return (
    <div>
      <div className="card">
        <h1>Experiments: change one thing, see what changes</h1>
        <p>Each intervention is a small, reversible edit applied to a real model on the HPC (head ablation, DeepStack off, Gated DeltaNet without forgetting, fewer experts, different image resolution, blank image, positions…). Pick a model × sample, then compare the baseline with a variant.</p>
        {!keys.length ? <Empty>No runs yet.</Empty> : (
          <div className="row">
            <label className="small">Model · sample · mode <select value={k} onChange={(e) => { setGk(e.target.value); setVid(null); }}>{keys.map((x) => <option key={x} value={x}>{x.replace(/__/g, " · ")} ({groups[x].length} runs)</option>)}</select></label>
            <label className="small">Variant <select value={variant?.id ?? ""} onChange={(e) => setVid(e.target.value)}>{runs.filter((r) => r.id !== base?.id).map((r) => <option key={r.id} value={r.id}>{r.intervention}</option>)}</select></label>
          </div>)}
        {runs.length === 1 && <div className="warn small" style={{ marginTop: 8 }}>Only the baseline exists for this sample. Variants come from <code>configs/experiments/interventions.yaml</code> on the HPC.</div>}
      </div>
      {diff.data && (
        <>
          <div className="grid2">
            {(["a", "b"] as const).map((s) => (
              <div className="card" key={s}>
                <h3>{s === "a" ? "Baseline" : "Variant"}: {diff.data[s].intervention.name}</h3>
                <p className="small">{diff.data[s].intervention.description}</p>
                <pre style={{ whiteSpace: "pre-wrap", maxHeight: 220, overflow: "auto" }}>{diff.data[s].reply.replace(/\n/g, "↵\n").slice(0, 3000) || "∅ (empty)"}</pre>
                <div className="small">exact-match score: <b>{diff.data[s].score ?? "needs judge"}</b> · T = {diff.data[s].T} · vision tokens {diff.data[s].n_vision_tokens}</div>
                {diff.data[s].first_step_top && <div className="small muted">first token candidates: {diff.data[s].first_step_top.slice(0, 5).map((x: any) => `${showTok(x[1])} ${(Math.exp(x[2]) * 100).toFixed(0)}%`).join(" · ")}</div>}
              </div>))}
          </div>
          <div className="card">
            <h3>Mean residual-stream norm per layer</h3>
            <LineChart width={1000} height={220} xLabel="layer" yLabel="mean ‖h‖" series={[{ name: "baseline", values: diff.data.a.residual_norm_mean, color: "var(--series-1)" }, { name: "variant", values: diff.data.b.residual_norm_mean, color: "var(--series-2)" }]} />
            {diff.data.residual_norm_rel_change && (<>
              <h3>Relative change of the residual stream on the shared prompt, per layer</h3>
              <p className="small">|‖h_variant‖ − ‖h_base‖| / ‖h_base‖ averaged over prompt tokens: where in depth the intervention starts to matter.</p>
              <LineChart width={1000} height={200} xLabel="layer" yLabel="relative change" series={[{ name: "relative change", values: diff.data.residual_norm_rel_change, color: "var(--series-8)" }]} />
            </>)}
          </div>
        </>)}
      <div className="card">
        <h2>Available interventions</h2>
        <table><thead><tr><th>name</th><th>what it does</th><th>families</th></tr></thead>
          <tbody>{(ivs.data ?? []).map((i) => <tr key={i.name}><td><code>{i.name}</code></td><td style={{ whiteSpace: "normal" }}>{i.description}</td><td>{i.families.join(", ")}</td></tr>)}</tbody></table>
        <Disclosure title="How to add runs (experiment YAML for the HPC)">
          <pre>{`# configs/experiments/my_ablation.yaml
models: [qwen3.5-2b, qwen3-vl-2b]
samples: [vlir-152, vstar-1]
modes: [direct]
policy: standard
interventions:
  - null                                   # baseline
  - {name: head_ablate, layer: 3, heads: [0, 1]}
  - {name: gdn_no_forget, layers: [0, 1, 2]}
  - {name: resolution, max_pixels: 100352}
# HPC: sbatch -p <partition> --array=0-$((N-1)) scripts/hpc/capture_array.slurm configs/experiments/my_ablation.yaml`}</pre>
        </Disclosure>
      </div>
    </div>
  );
}
