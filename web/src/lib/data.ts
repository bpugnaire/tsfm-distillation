export type Role = "student" | "teacher" | "comparator" | "baseline";
export interface GalleryModel { name: string; role: Role; n_params: number; size_mb: number | null; latency_ms: number | null; tails_extrapolated: boolean; fallbacks: string[]; }
export interface GalleryShape { name: string; description: string; period: number | null; context: (number | null)[]; future: number[]; }
export interface Cell { q: number[][]; wql: number | null; mase: number | null; }
export interface Gallery { levels: number[]; shapes: GalleryShape[]; models: GalleryModel[]; cells: Record<string, Record<string, Cell>>; hardware: { cpu: string }; }
export interface SummaryRow { model: string; role: Role; n_params: number; rel_wql: number; rel_wql_ci: [number, number]; rel_mase: number; mean_cov80: number; mean_cov98: number; tails_extrapolated: boolean; latency_ms_t1_b1: number | null; latency_ms_t8_b32: number | null; n_windows: number; n_clusters: number; }
export interface Results {
  suites: Record<string, { rows: SummaryRow[]; git_sha: string }>;
  per_shape: Record<string, Record<string, Record<string, number>> | null>;
  splits: { train_end: string; val_start: string; val_end: string; fresh_start: string; releases: Record<string, string> };
  export: Record<string, { n_params: number; fp32: ParityRep; int8: ParityRep }>;
  throughput: Record<string, { windows_per_s?: number; device: string; error?: string }>;
  crps_bias: { exact: number; equal_weights: Record<string, number>; mass_weights: Record<string, number> };
}
export interface ParityRep { max_rel_diff: number; p99_rel_diff: number; size_mb: number; }

const base = import.meta.env.BASE_URL;
export async function loadJSON<T>(name: string): Promise<T> {
  const r = await fetch(`${base}data/${name}`);
  if (!r.ok) throw new Error(`failed to load ${name}: ${r.status}`);
  return (await r.json()) as T;
}

export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
export const roleColor = (r: Role) => cssVar(`--${r}`);

export function fmtParams(n: number): string {
  if (!n) return "–";
  return n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)}M` : `${(n / 1e3).toFixed(0)}k`;
}
export const isPreview = (name: string) => name.includes("preview");
