// Types of the JSON files written by scripts/build_web_data.py, and small shared helpers.
export type Role = "student" | "teacher" | "comparator" | "baseline";
export type Ratio = [number, number, number]; // point, 95% CI low, high

export interface Manifest {
  onnx: string; n_params: number; size_mb: { fp32: number; int8: number }; parity_p99: { fp32: number; int8: number };
  cpu_ms_onnx_t1_b1: { fp32: number; int8: number }; context: number; horizon: number;
  teacher: { name: string; label: string; n_params: number }; source: string[];
}
export interface Shape { name: string; label: string; description: string; context: (number | null)[]; future: number[]; teacher_median: number[]; student_rel_wql: number }
export interface Shapes { shapes: Shape[]; source: string }
export interface Row { model: string; label: string; role: Role; rel_wql: number; ci: [number, number]; cov80: number; n_params: number; ms: number | null }
export interface Suite { rows: Row[]; n_windows: number; n_series: number; source: string[] }
export interface GiftRow { model: string; label: string; role: Role; crps: number; mase: number | null }
export interface Contrast { suite: string; name: string; ratio: Ratio; source: string }
export interface Finding { group: "worth" | "recipe" | "ship"; title: string; versus: string; text: string; takeaway: string; contrasts: Contrast[] }
export interface Results {
  suites: Record<string, Suite>;
  gift: { n_configs: number; rank: number; n_models: number; rows: GiftRow[]; source: string[] };
  findings: Finding[];
  splits: { train_end: string; fresh_start: string; releases: Record<string, string> };
}
export interface Snippet { file: string; line: number; code: string; lang: "python" | "typescript" }

const base = import.meta.env.BASE_URL;
export async function loadJSON<T>(name: string): Promise<T> {
  const r = await fetch(`${base}data/${name}`);
  if (!r.ok) throw new Error(`failed to load ${name}: ${r.status}`);
  return (await r.json()) as T;
}
export const cssVar = (name: string): string => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export const roleColor = (r: Role): string => cssVar(`--${r}`);

export function fmtParams(n: number): string {
  if (!n) return "–";
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  return n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)}M` : `${(n / 1e3).toFixed(0)}k`;
}

/** Paired ratio a/b in words. Significant only if the 95% CI excludes 1. */
export function ratioWords(r: Ratio): { text: string; cls: "good" | "bad" | "tie" } {
  const [p, lo, hi] = r;
  if (lo <= 1 && hi >= 1) return { text: "no detectable difference", cls: "tie" };
  if (p < 1) return { text: `${Math.round((1 - p) * 100)}% lower error`, cls: "good" };
  return { text: p < 1.5 ? `${Math.round((p - 1) * 100)}% higher error` : `${p.toFixed(1)}× the error`, cls: "bad" };
}

const tip = (): HTMLElement => {
  let t = document.getElementById("tip");
  if (!t) { t = document.createElement("div"); t.id = "tip"; t.className = "tooltip"; document.body.append(t); }
  return t;
};
export function showTip(e: MouseEvent, html: string): void {
  const t = tip(); t.innerHTML = html; t.style.display = "block";
  t.style.left = `${Math.min(e.clientX + 14, innerWidth - t.offsetWidth - 8)}px`; t.style.top = `${e.clientY + 14}px`;
}
export const hideTip = (): void => { tip().style.display = "none"; };
