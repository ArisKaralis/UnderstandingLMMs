import { useEffect, useRef } from "react";
import { useScale } from "./charts";
import { useTip } from "./ui";
import { fmtNum } from "../lib/format";

/**
 * Image with a grid heat overlay. `values` is in raster order (gh × gw) — callers convert from the
 * ViT's merge-window order using tokens.images[k].raster_order.
 */
export function ImageHeat({ src, gh, gw, values, width = 420, alpha = 0.55, showGrid = false, mergeGrid = 0, onCell, selected, label = "value" }: {
  src: string; gh: number; gw: number; values?: number[] | null; width?: number; alpha?: number; showGrid?: boolean;
  mergeGrid?: number; onCell?: (r: number, c: number) => void; selected?: [number, number] | null; label?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const height = Math.round((width * gh) / gw);
  const vs = values ?? [];
  const sorted = [...vs].sort((a, b) => a - b);
  const color = useScale("seq", [sorted[0] ?? 0, sorted[Math.floor(sorted.length * 0.995)] ?? 1]);
  const { tip, show, hide } = useTip();
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = width * dpr; c.height = height * dpr;
    const ctx = c.getContext("2d")!; ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, height);
    const cw = width / gw, ch = height / gh;
    if (values) {
      ctx.globalAlpha = alpha;
      for (let r = 0; r < gh; r++) for (let q = 0; q < gw; q++) { ctx.fillStyle = color(values[r * gw + q]); ctx.fillRect(q * cw, r * ch, Math.ceil(cw), Math.ceil(ch)); }
      ctx.globalAlpha = 1;
    }
    if (showGrid || mergeGrid) {
      ctx.strokeStyle = "rgba(255,255,255,0.45)"; ctx.lineWidth = 0.5;
      if (showGrid) { for (let q = 0; q <= gw; q++) { ctx.beginPath(); ctx.moveTo(q * cw, 0); ctx.lineTo(q * cw, height); ctx.stroke(); } for (let r = 0; r <= gh; r++) { ctx.beginPath(); ctx.moveTo(0, r * ch); ctx.lineTo(width, r * ch); ctx.stroke(); } }
      if (mergeGrid) { ctx.strokeStyle = "rgba(235,104,52,0.9)"; ctx.lineWidth = 1.2; for (let q = 0; q <= gw; q += mergeGrid) { ctx.beginPath(); ctx.moveTo(q * cw, 0); ctx.lineTo(q * cw, height); ctx.stroke(); } for (let r = 0; r <= gh; r += mergeGrid) { ctx.beginPath(); ctx.moveTo(0, r * ch); ctx.lineTo(width, r * ch); ctx.stroke(); } }
    }
    if (selected) { ctx.strokeStyle = "#eb6834"; ctx.lineWidth = 2.5; ctx.strokeRect(selected[1] * cw, selected[0] * ch, cw, ch); }
  }, [values, gh, gw, width, height, alpha, showGrid, mergeGrid, selected, color]);
  const pos = (e: React.MouseEvent) => {
    const r = (e.target as HTMLElement).getBoundingClientRect();
    return [Math.min(gh - 1, Math.floor(((e.clientY - r.top) / r.height) * gh)), Math.min(gw - 1, Math.floor(((e.clientX - r.left) / r.width) * gw))] as const;
  };
  return (
    <div style={{ position: "relative", width, height }}>
      <img src={src} style={{ width, height, display: "block", borderRadius: 4 }} />
      <canvas ref={ref} style={{ position: "absolute", inset: 0, width, height, cursor: onCell ? "pointer" : "crosshair" }}
        onMouseMove={(e) => { const [r, c] = pos(e); show(e, <span>patch row {r}, col {c}{values ? <> · <b>{label} {fmtNum(values[r * gw + c], 4)}</b></> : null}</span>); }}
        onMouseLeave={hide} onClick={(e) => { const [r, c] = pos(e); onCell?.(r, c); }} />
      {tip}
    </div>
  );
}

/** ViT sequence order → raster order */
export function toRaster(seqVals: number[], rasterOrder: number[], offset = 0): number[] {
  const out = new Array(rasterOrder.length).fill(0);
  rasterOrder.forEach((r, k) => { out[r] = seqVals[offset + k]; });
  return out;
}

/** merged-token values (one per 2×2 window, in LLM token order) → per-patch raster grid */
export function mergedToRaster(vals: number[], gh: number, gw: number, m: number): number[] {
  const out = new Array(gh * gw).fill(0);
  const mw = gw / m;
  for (let r = 0; r < gh; r++) for (let c = 0; c < gw; c++) out[r * gw + c] = vals[Math.floor(r / m) * mw + Math.floor(c / m)];
  return out;
}
