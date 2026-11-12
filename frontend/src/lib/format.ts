export function fmtNum(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "–";
  const a = Math.abs(n);
  if (a >= 1e12) return (n / 1e12).toFixed(digits) + "T";
  if (a >= 1e9) return (n / 1e9).toFixed(digits) + "B";
  if (a >= 1e6) return (n / 1e6).toFixed(digits) + "M";
  if (a >= 1e3) return (n / 1e3).toFixed(digits) + "K";
  if (a !== 0 && a < 0.01) return n.toExponential(1);
  return Number.isInteger(n) ? String(n) : n.toFixed(digits);
}
export function fmtBytes(b: number): string {
  if (b >= 2 ** 30) return (b / 2 ** 30).toFixed(2) + " GiB";
  if (b >= 2 ** 20) return (b / 2 ** 20).toFixed(1) + " MiB";
  if (b >= 2 ** 10) return (b / 2 ** 10).toFixed(1) + " KiB";
  return b + " B";
}
export function fmtFlops(f: number): string {
  return fmtNum(f, 2).replace(/([KMBT])$/, (m) => ({ K: " K", M: " M", B: " G", T: " T" } as any)[m]) + "FLOP";
}
export const prod = (xs: number[]) => xs.reduce((a, b) => a * b, 1);
export const showTok = (s: string) => s.replace(/\n/g, "↵").replace(/ /g, "·") || "∅";
