// Forecasts longer than the model's 64 steps, by rolling it out block by block (port of tinycast/models/rollout.py,
// "paths" mode, the better of the two modes measured in results/long_horizon/).
//
// Block 1 is the model's own forecast. For later blocks we draw P sample paths: within a block, a path reads its value
// from the forecast quantiles through a Gaussian copula whose latent noise is AR(1) over time (rho 0.9), so a path that
// is high stays high for a while. Each path is appended to its own context and forecast again. The quantiles shown past
// step 64 are taken across the paths. Feeding back only the median would make the bands far too narrow (80% interval
// coverage 0.30 at 512 steps on the shapes, against 0.72 with paths).
import { LEVELS, type Student } from "./ort";

function mulberry32(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(r: () => number): number { return Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r()); }
function ncdf(z: number): number { // Abramowitz-Stegun 26.2.17, |error| < 7.5e-8
  const t = 1 / (1 + 0.2316419 * Math.abs(z)), d = 0.3989423 * Math.exp(-z * z / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return z > 0 ? 1 - p : p;
}
function quantile(sorted: number[], u: number): number { // numpy's default (linear) rule
  const h = (sorted.length - 1) * u, lo = Math.floor(h);
  return sorted[lo] + (h - lo) * ((sorted[Math.min(lo + 1, sorted.length - 1)]) - sorted[lo]);
}

/** Fast preview: feed back the median only (one pass per block). Bands past the first block would be far too narrow,
 *  so only the median is returned for them (q rows hold NaN there). */
export async function rolloutMedian(model: Student, ctx: number[], horizon: number):
  Promise<{ q: number[][]; ms: number; msFirst: number; passes: number }> {
  const first = await model.forecast(ctx);
  const i50 = LEVELS.indexOf(0.5);
  const q = first.q.map((r) => r.slice(0, horizon));
  let ms = first.ms, passes = 1, c = [...ctx, ...first.q[i50]];
  while (q[0].length < horizon) {
    const r = await model.forecast(c);
    ms += r.ms; passes++; c = [...c, ...r.q[i50]];
    LEVELS.forEach((_, k) => q[k].push(...(k === i50 ? r.q[i50] : r.q[i50].map(() => NaN))));
  }
  return { q: q.map((r) => r.slice(0, horizon)), ms, msFirst: first.ms, passes };
}

/** stale(): checked between passes; when it returns true the rollout stops and returns null (newer input arrived). */
export async function rollout(model: Student, ctx: number[], horizon: number, P = 64, rho = 0.9, stale: () => boolean = () => false):
  Promise<{ q: number[][]; ms: number; msFirst: number; passes: number } | null> {
  const first = await model.forecast(ctx);
  const B = first.q[0].length;
  if (horizon <= B) return { q: first.q.map((r) => r.slice(0, horizon)), ms: first.ms, msFirst: first.ms, passes: 1 };
  const rand = mulberry32(12345); // fixed seed: the same drawing gives the same forecast
  const K = LEVELS.length, a = Math.sqrt(1 - rho * rho);
  const z = Array.from({ length: P }, () => gauss(rand));
  const paths: number[][] = Array.from({ length: P }, () => []);
  const draw = (q: number[][], p: number) => {
    for (let t = 0; t < B; t++) {
      z[p] = rho * z[p] + a * gauss(rand);
      const u = Math.min(Math.max(ncdf(z[p]), LEVELS[0]), LEVELS[K - 1]);
      let k = 0; while (k < K - 2 && LEVELS[k + 1] < u) k++;
      const w = (u - LEVELS[k]) / (LEVELS[k + 1] - LEVELS[k]);
      paths[p].push(q[k][t] + w * (q[k + 1][t] - q[k][t]));
    }
  };
  for (let p = 0; p < P; p++) draw(first.q, p);
  let ms = first.ms, passes = 1;
  while (paths[0].length < horizon) {
    await new Promise((res) => setTimeout(res, 0)); // let pointer events through between passes
    if (stale()) return null;
    const r = await model.forecastBatch(paths.map((pt) => [...ctx, ...pt]));
    ms += r.ms; passes++;
    for (let p = 0; p < P; p++) draw(r.q[p], p);
  }
  const q = LEVELS.map((_, k) => first.q[k].slice()); // block 1: the model's own quantiles
  for (let t = B; t < horizon; t++) {
    const col = paths.map((pt) => pt[t]).sort((x, y) => x - y);
    // 1% / 99% from 32 paths would be the sample min / max: too noisy to draw, so the 98% band stops at the first block
    LEVELS.forEach((u, k) => q[k].push(u < 0.05 || u > 0.95 ? NaN : quantile(col, u)));
  }
  return { q, ms, msFirst: first.ms, passes };
}
