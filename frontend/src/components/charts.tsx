import * as d3 from "d3";
import { ReactNode, useEffect, useMemo, useRef } from "react";
import { useCssVar, useTip } from "./ui";
import { fmtNum } from "../lib/format";

/** Sequential (one hue, light→dark) or diverging (blue ↔ grey ↔ red) colour scale from the reference palette. */
export function useScale(kind: "seq" | "div", domain: [number, number]) {
  const neutral = useCssVar("--div-neutral");
  const surface = useCssVar("--surface-1");
  return useMemo(() => {
    if (kind === "div") {
      const m = Math.max(Math.abs(domain[0]), Math.abs(domain[1])) || 1;
      return d3.scaleLinear<string>().domain([-m, 0, m]).range(["#1c5cab", neutral || "#f0efec", "#c0392b"]).interpolate(d3.interpolateLab).clamp(true);
    }
    const dark = (surface || "").toLowerCase() === "#1a1a19";
    return d3.scaleLinear<string>().domain(domain).range(dark ? ["#1a1a19", "#86b6ef"] : ["#cde2fb", "#0d366b"]).interpolate(d3.interpolateLab).clamp(true);
  }, [kind, domain[0], domain[1], neutral, surface]);
}

type HeatProps = {
  data: ArrayLike<number>; rows: number; cols: number; width?: number; height?: number; kind?: "seq" | "div";
  domain?: [number, number]; rowLabel?: (i: number) => string; colLabel?: (j: number) => string; valueLabel?: string;
  onCell?: (i: number, j: number) => void; highlightCol?: number | null; log?: boolean; title?: string;
};

/** Canvas heatmap with per-cell hover tooltip (rows × cols, row-major). */
export function Heatmap({ data, rows, cols, width = 640, height, kind = "seq", domain, rowLabel, colLabel, valueLabel = "value", onCell, highlightCol, log, title }: HeatProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const h = height ?? Math.max(60, Math.min(420, rows * 8));
  const vals = useMemo(() => {
    const v = Array.from(data as ArrayLike<number>);
    return log ? v.map((x) => Math.log10(Math.max(x, 1e-6))) : v;
  }, [data, log]);
  const dom = useMemo<[number, number]>(() => {
    if (domain) return domain;
    const s = [...vals].filter(Number.isFinite).sort((a, b) => a - b);
    return [s[Math.floor(s.length * 0.01)] ?? 0, s[Math.floor(s.length * 0.99)] ?? 1];
  }, [vals, domain]);
  const color = useScale(kind, dom);
  const { tip, show, hide } = useTip();
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = width * dpr; c.height = h * dpr;
    const ctx = c.getContext("2d")!; ctx.scale(dpr, dpr);
    const cw = width / cols, ch = h / rows;
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      ctx.fillStyle = color(vals[i * cols + j]);
      ctx.fillRect(j * cw, i * ch, Math.ceil(cw), Math.ceil(ch));
    }
    if (highlightCol !== undefined && highlightCol !== null) {
      ctx.strokeStyle = "#eb6834"; ctx.lineWidth = 2; ctx.strokeRect(highlightCol * cw, 0, Math.max(cw, 2), h);
    }
  }, [vals, rows, cols, width, h, color, highlightCol]);
  const pick = (e: React.MouseEvent) => {
    const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
    const j = Math.min(cols - 1, Math.floor(((e.clientX - r.left) / r.width) * cols));
    const i = Math.min(rows - 1, Math.floor(((e.clientY - r.top) / r.height) * rows));
    return [i, j] as const;
  };
  return (
    <div>
      {title && <h4>{title}</h4>}
      <canvas ref={ref} style={{ width, height: h, cursor: onCell ? "pointer" : "crosshair", borderRadius: 4, display: "block" }}
        onMouseMove={(e) => { const [i, j] = pick(e); const v = (data as any)[i * cols + j];
          show(e, <div>{rowLabel ? rowLabel(i) : `row ${i}`} · {colLabel ? colLabel(j) : `col ${j}`}<br /><b>{valueLabel}: {fmtNum(v, 4)}</b></div>); }}
        onMouseLeave={hide} onClick={(e) => { const [i, j] = pick(e); onCell?.(i, j); }} />
      <ColorBar color={color} dom={dom} log={log} label={valueLabel} />
      {tip}
    </div>
  );
}

function ColorBar({ color, dom, log, label }: { color: (v: number) => string; dom: [number, number]; log?: boolean; label: string }) {
  const stops = d3.range(0, 1.0001, 0.1).map((t) => color(dom[0] + t * (dom[1] - dom[0])));
  const f = (v: number) => fmtNum(log ? 10 ** v : v, 3);
  return (
    <div className="row small muted" style={{ marginTop: 4 }}>
      <span>{f(dom[0])}</span>
      <span style={{ width: 140, height: 8, borderRadius: 4, background: `linear-gradient(90deg, ${stops.join(",")})` }} />
      <span>{f(dom[1])}</span><span>{label}{log ? " (log)" : ""}</span>
    </div>
  );
}

export type Series = { name: string; values: number[]; color: string };

/** Line chart with crosshair tooltip; one y-axis only. */
export function LineChart({ series, width = 640, height = 220, xLabel = "", yLabel = "", x0 = 0, marks, yDomain, xTick }: {
  series: Series[]; width?: number; height?: number; xLabel?: string; yLabel?: string; x0?: number;
  marks?: { x: number; label: string }[]; yDomain?: [number, number]; xTick?: (i: number) => string;
}) {
  const m = { l: 48, r: 12, t: 10, b: 30 };
  const n = Math.max(...series.map((s) => s.values.length), 1);
  const all = series.flatMap((s) => s.values).filter(Number.isFinite);
  const x = d3.scaleLinear().domain([x0, x0 + n - 1]).range([m.l, width - m.r]);
  const y = d3.scaleLinear().domain(yDomain ?? (d3.extent(all) as [number, number])).nice().range([height - m.b, m.t]);
  const { tip, show, hide } = useTip();
  const ref = useRef<SVGSVGElement>(null);
  const hover = useRef<number | null>(null);
  const grid = useCssVar("--grid"), axis = useCssVar("--text-muted");
  const line = d3.line<number>().defined(Number.isFinite).x((_, i) => x(x0 + i)).y((v) => y(v));
  const onMove = (e: React.MouseEvent) => {
    const r = ref.current!.getBoundingClientRect();
    const i = Math.round(x.invert(e.clientX - r.left)) - x0;
    if (i < 0 || i >= n) return hide();
    hover.current = i;
    show(e, <div><b>{xTick ? xTick(i + x0) : `${xLabel} ${i + x0}`}</b>{series.map((s) => (
      <div key={s.name}><span className="sw" style={{ display: "inline-block", width: 8, height: 8, background: s.color, marginRight: 4 }} />{s.name}: {fmtNum(s.values[i], 4)}</div>))}</div>);
  };
  return (
    <div style={{ position: "relative" }}>
      <svg ref={ref} width={width} height={height} onMouseMove={onMove} onMouseLeave={hide}>
        {y.ticks(4).map((t) => (<g key={t}><line x1={m.l} x2={width - m.r} y1={y(t)} y2={y(t)} stroke={grid} /><text x={m.l - 6} y={y(t)} dy="0.32em" textAnchor="end" fontSize={10} fill={axis}>{fmtNum(t, 2)}</text></g>))}
        {x.ticks(8).filter(Number.isInteger).map((t) => (<text key={t} x={x(t)} y={height - m.b + 14} textAnchor="middle" fontSize={10} fill={axis}>{xTick ? xTick(t) : t}</text>))}
        <text x={width - m.r} y={height - 4} textAnchor="end" fontSize={10} fill={axis}>{xLabel}</text>
        <text x={4} y={m.t + 2} fontSize={10} fill={axis}>{yLabel}</text>
        {marks?.map((mk) => (<g key={mk.label + mk.x}><line x1={x(mk.x)} x2={x(mk.x)} y1={m.t} y2={height - m.b} stroke={axis} strokeDasharray="3 3" /><text x={x(mk.x) + 3} y={m.t + 10} fontSize={10} fill={axis}>{mk.label}</text></g>))}
        {series.map((s) => (<path key={s.name} d={line(s.values) ?? ""} fill="none" stroke={s.color} strokeWidth={2} />))}
      </svg>
      {series.length > 1 && <div className="legend">{series.map((s) => <span key={s.name}><span className="sw" style={{ background: s.color }} />{s.name}</span>)}</div>}
      {tip}
    </div>
  );
}

/** Horizontal bars (sorted by caller), with hover. */
export function Bars({ items, width = 360, color = "var(--series-1)", fmt = (v: number) => fmtNum(v, 3) }: {
  items: { label: ReactNode; value: number; key: string; hint?: string }[]; width?: number; color?: string; fmt?: (v: number) => string;
}) {
  const max = Math.max(...items.map((i) => Math.abs(i.value)), 1e-9);
  const { tip, show, hide } = useTip();
  return (
    <div>
      {items.map((i) => (
        <div key={i.key} className="row small" style={{ gap: 6, margin: "2px 0" }} onMouseMove={(e) => show(e, <span>{i.hint ?? ""} <b>{fmt(i.value)}</b></span>)} onMouseLeave={hide}>
          <span style={{ width: 110, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{i.label}</span>
          <span style={{ width: (width - 170) * Math.abs(i.value) / max, minWidth: 2, height: 10, background: color, borderRadius: "0 4px 4px 0" }} />
          <span className="muted">{fmt(i.value)}</span>
        </div>
      ))}
      {tip}
    </div>
  );
}
