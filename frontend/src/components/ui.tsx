import katex from "katex";
import "katex/dist/katex.min.css";
import { ReactNode, useMemo, useState, useEffect } from "react";
import type { ShapeDim } from "../lib/api";
import { fmtBytes, prod } from "../lib/format";

export function TeX({ tex, block = false }: { tex: string; block?: boolean }) {
  const html = useMemo(() => {
    try { return katex.renderToString(tex, { displayMode: block, throwOnError: false }); } catch { return tex; }
  }, [tex, block]);
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

export function Disclosure({ title, children, open = false }: { title: ReactNode; children: ReactNode; open?: boolean }) {
  return (
    <details className="disclosure" open={open}>
      <summary>{title}</summary>
      <div className="body">{children}</div>
    </details>
  );
}

export function Shape({ s, bytesPer = 2 }: { s: ShapeDim[]; bytesPer?: number }) {
  const n = prod(s.map((d) => d[1]));
  return (
    <span className="shape">
      [{s.map((d, i) => (<span key={i}>{i > 0 && ", "}<span className="sym">{d[0]}=</span>{d[1].toLocaleString()}</span>))}]
      <span className="muted"> · {n.toLocaleString()} el · {fmtBytes(n * bytesPer)} bf16</span>
    </span>
  );
}

export function ShapeTable({ title, m }: { title: string; m: Record<string, ShapeDim[]> }) {
  const e = Object.entries(m);
  if (!e.length) return null;
  return (
    <div>
      <h4>{title}</h4>
      <table><tbody>{e.map(([k, s]) => (<tr key={k}><td><code>{k}</code></td><td><Shape s={s} /></td></tr>))}</tbody></table>
    </div>
  );
}

/** Fixed-position tooltip that follows the pointer. */
export function useTip() {
  const [tip, setTip] = useState<{ x: number; y: number; content: ReactNode } | null>(null);
  const el = tip ? <div className="tooltip" style={{ left: tip.x + 14, top: tip.y + 14 }}>{tip.content}</div> : null;
  return { tip: el, show: (e: { clientX: number; clientY: number }, content: ReactNode) => setTip({ x: e.clientX, y: e.clientY, content }), hide: () => setTip(null) };
}

export function Legend({ items }: { items: { label: string; color: string; pattern?: boolean }[] }) {
  return (
    <div className="legend">
      {items.map((i) => (
        <span key={i.label}><span className="sw" style={{ background: i.pattern ? `repeating-linear-gradient(45deg, ${i.color} 0 2px, transparent 2px 5px)` : i.color, border: i.pattern ? `1px solid ${i.color}` : undefined }} />{i.label}</span>
      ))}
    </div>
  );
}

export function Loading({ what = "" }: { what?: string }) { return <div className="empty">Loading {what}…</div>; }
export function Empty({ children }: { children: ReactNode }) { return <div className="empty">{children}</div>; }

export function useCssVar(name: string): string {
  const [v, setV] = useState(() => getComputedStyle(document.documentElement).getPropertyValue(name).trim());
  useEffect(() => {
    const obs = new MutationObserver(() => setV(getComputedStyle(document.documentElement).getPropertyValue(name).trim()));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => obs.disconnect();
  }, [name]);
  return v;
}
