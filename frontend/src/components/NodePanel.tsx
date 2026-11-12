import type { GNode } from "../lib/api";
import { useTracePart } from "../lib/api";
import { fmtFlops, fmtNum } from "../lib/format";
import { Disclosure, ShapeTable, TeX } from "./ui";
import { glossaryFor } from "../content/glossary";

/** Side panel for one architecture node: explanation first, tensor/op details behind closed drop-downs. */
export function NodePanel({ node, traceId }: { node: GNode | null; traceId?: string | null }) {
  const stats = useTracePart<any[]>(traceId ?? null, "stats");
  if (!node) return <div className="card muted">Click any block to see what it does, its tensor shapes and operations.</div>;
  const rows = (stats.data ?? []).filter((r) => r.node === node.id);
  const g = glossaryFor(node.kind);
  return (
    <div className="card">
      <div className="row"><span className="pill">{node.kind}</span><span className="pill">{node.level}</span>{node.ref && <span className="small muted">{node.ref}</span>}</div>
      <h2 style={{ marginTop: 6 }}>{node.label}</h2>
      {node.info && <p>{node.info}</p>}
      {!node.info && g && <p>{g}</p>}
      {node.eq && <div style={{ overflowX: "auto", padding: "4px 0" }}><TeX tex={node.eq} block /></div>}
      <div className="kpi" style={{ margin: "8px 0" }}>
        <div><b>{fmtNum(node.params)}</b><span>parameters</span></div>
        {node.flops > 0 && <div><b>{fmtFlops(node.flops)}</b><span>for this sequence</span></div>}
      </div>
      <Disclosure title="Tensor details (shapes)">
        <ShapeTable title="Inputs" m={node.inputs} />
        <ShapeTable title="Outputs" m={node.outputs} />
        <ShapeTable title="Weights" m={node.weights} />
        {Object.keys(node.meta ?? {}).length > 0 && <pre>{JSON.stringify(node.meta, null, 1)}</pre>}
      </Disclosure>
      {node.op && (<Disclosure title="Operation (pseudo-code)"><pre>{node.op}</pre></Disclosure>)}
      {traceId && (
        <Disclosure title={`Live values from this trace (${rows.length} recorded output${rows.length === 1 ? "" : "s"})`}>
          {rows.length === 0 && <div className="muted small">Not recorded for this node (policy or not a hooked module). Children may have values.</div>}
          {rows.map((r, k) => (<StatRow key={k} r={r} />))}
        </Disclosure>
      )}
    </div>
  );
}

function StatRow({ r }: { r: any }) {
  const max = Math.max(...r.hist);
  return (
    <div style={{ margin: "6px 0 10px" }}>
      <div className="small"><code>{r.name}</code> <span className="muted">{JSON.stringify(r.shape)} {r.dtype}</span></div>
      <table><tbody><tr>
        <td>mean {fmtNum(r.mean, 4)}</td><td>std {fmtNum(r.std, 4)}</td><td>min {fmtNum(r.min, 3)}</td><td>max {fmtNum(r.max, 3)}</td><td>‖x‖ {fmtNum(r.l2, 3)}</td><td>zeros {(r.frac_zero * 100).toFixed(1)}%</td>
      </tr></tbody></table>
      <svg width={260} height={40} aria-label="value histogram">
        {r.hist.map((h: number, i: number) => (<rect key={i} x={i * (260 / r.hist.length)} y={40 - (36 * h) / max} width={260 / r.hist.length - 1} height={(36 * h) / max} rx={1} fill="var(--series-1)" />))}
      </svg>
      <div className="small muted">histogram over [{fmtNum(r.hist_lo, 3)}, {fmtNum(r.hist_hi, 3)}]</div>
    </div>
  );
}
