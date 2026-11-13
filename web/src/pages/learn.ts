// Shared script of every Learn page: sidebar, glossary tooltips, code snippets from the repo, widgets.
// Widgets mount only if their element exists on the page. Everything is seeded: all readers see the same thing.
import { scaleLinear } from "d3-scale";
import { select, type Selection } from "d3-selection";
import { area, line } from "d3-shape";
import { header } from "../lib/nav";
import { cssVar, loadJSON, type Gallery, type Results } from "../lib/data";
import { drawFan } from "../lib/fan";
import { GLOSS, GLOSSARY } from "../lib/glossary";

header();
const $ = (id: string) => document.getElementById(id);
type SVG = Selection<SVGSVGElement, unknown, HTMLElement, unknown>;

// ------------------------------------------------------------------ sidebar
const CHAPTERS: [string, string][] = [
  ["learn.html", "Overview & learning path"],
  ["learn-1-basics.html", "1 · Forecasting fundamentals"],
  ["learn-2-scoring.html", "2 · Probabilistic forecasts & scores"],
  ["learn-3-tsfm.html", "3 · How foundation models work"],
  ["learn-4-distillation.html", "4 · Knowledge distillation"],
  ["learn-5-deployment.html", "5 · Compression & the browser"],
  ["learn-6-ablations.html", "6 · Running an ablation study"],
  ["glossary.html", "Glossary"],
];
{
  const aside = $("sidebar");
  if (aside) {
    const here = location.pathname.split("/").pop() || "learn.html";
    const onPage = [...document.querySelectorAll("article h2[id]")].map((h) => `<li><a href="#${h.id}">${h.textContent}</a></li>`).join("");
    aside.innerHTML = `<div class="toc-sec">Course</div><ol style="list-style:none;padding-left:0">${CHAPTERS.map(([f, t]) =>
      `<li><a href="${f}"${f === here ? ' aria-current="page"' : ""}>${t}</a></li>`).join("")}</ol>` +
      (onPage ? `<div class="toc-sec">On this page</div><ol>${onPage}</ol>` : "");
    const i = CHAPTERS.findIndex(([f]) => f === here);
    const nav = $("chapter-nav");
    if (nav && i >= 0) nav.innerHTML = `<span>${i > 0 ? `← <a href="${CHAPTERS[i - 1][0]}">${CHAPTERS[i - 1][1]}</a>` : ""}</span>` +
      `<span>${i < CHAPTERS.length - 1 ? `<a href="${CHAPTERS[i + 1][0]}">${CHAPTERS[i + 1][1]}</a> →` : ""}</span>`;
  }
}

// ------------------------------------------------------------------ glossary tooltips + glossary page
for (const el of document.querySelectorAll<HTMLElement>(".term[data-g]")) {
  const t = GLOSS[el.dataset.g!];
  if (t) el.title = `${t.term}: ${t.def}`;
}
{
  const g = $("glossary-list"), q = $("glossary-search") as HTMLInputElement | null;
  if (g && q) {
    const names = ["", "Fundamentals", "Scoring", "Foundation models", "Distillation", "Deployment", "Experiments"];
    const render = () => {
      const s = q.value.toLowerCase();
      const hits = GLOSSARY.filter((t) => !s || (t.term + " " + t.def).toLowerCase().includes(s)).sort((a, b) => a.term.localeCompare(b.term));
      g.innerHTML = hits.length ? hits.map((t) => `<dt id="g-${t.key}">${t.term} <a class="tag" href="${CHAPTERS[t.ch][0]}">ch. ${t.ch} · ${names[t.ch]}</a></dt><dd>${t.def}</dd>`).join("") : `<p class="muted">No term matches.</p>`;
    };
    q.addEventListener("input", render);
    render();
  }
}

// ------------------------------------------------------------------ code snippets extracted from the repo
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
const codeFigs = document.querySelectorAll<HTMLElement>("figure.code[data-ref]");
if (codeFigs.length) {
  loadJSON<Record<string, { file: string; line: number; code: string }>>("code.json").then((code) => {
    for (const f of codeFigs) {
      const c = code[f.dataset.ref!];
      if (!c) { f.innerHTML = `<figcaption>missing snippet ${f.dataset.ref}</figcaption>`; continue; }
      f.innerHTML = `<figcaption><code>${c.file}:${c.line}</code></figcaption><pre><code>${esc(c.code)}</code></pre>`;
    }
  });
}

// ------------------------------------------------------------------ helpers
function mulberry32(a: number) { return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function gaussian(rnd: () => number) { return () => Math.sqrt(-2 * Math.log(rnd() || 1e-12)) * Math.cos(2 * Math.PI * rnd()); }
const erf = (x: number) => { const t = 1 / (1 + 0.3275911 * Math.abs(x)); const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
const Phi = (x: number) => 0.5 * (1 + erf(x / Math.SQRT2));
const phi = (x: number) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
function svgOf(id: string, W: number, H: number): SVG | null {
  const el = $(id);
  if (!el) return null;
  return select(el as unknown as SVGSVGElement).attr("viewBox", `0 0 ${W} ${H}`).attr("width", "100%") as unknown as SVG;
}
function grid(svg: SVG, y: (v: number) => number, ticks: number[], x0: number, x1: number, fmt = (v: number) => String(v)) {
  for (const t of ticks) {
    svg.append("line").attr("x1", x0).attr("x2", x1).attr("y1", y(t)).attr("y2", y(t)).attr("stroke", cssVar("--grid"));
    svg.append("text").attr("x", x0 - 5).attr("y", y(t) + 4).attr("text-anchor", "end").text(fmt(t));
  }
}
function stats(id: string, items: [string, string][]) {
  const el = $(id);
  if (el) el.innerHTML = items.map(([k, v]) => `<span class="s">${k} <b>${v}</b></span>`).join("");
}
const ctrl = (id: string) => $(id) as HTMLInputElement;

// ------------------------------------------------------------------ ch1: anatomy of a series + backtest + ACF
function acfOf(x: number[], maxLag: number): number[] {
  const n = x.length, a = x[0], b = x[n - 1];
  const d = x.map((v, i) => v - (a + ((b - a) * i) / (n - 1)));
  const mean = d.reduce((s, v) => s + v, 0) / n;
  const c = d.map((v) => v - mean), den = c.reduce((s, v) => s + v * v, 0) || 1;
  return Array.from({ length: maxLag + 1 }, (_, k) => { let s = 0; for (let i = 0; i < n - k; i++) s += c[i] * c[i + k]; return s / den; });
}
function seasonFromAcf(acf: number[], thr = 0.3): number {
  const peaks: number[] = [];
  for (let k = 2; k < acf.length - 1; k++) if (acf[k] > acf[k - 1] && acf[k] >= acf[k + 1] && acf[k] > thr) peaks.push(k);
  if (!peaks.length) return 1;
  const best = Math.max(...peaks.map((k) => acf[k]));
  return Math.min(...peaks.filter((k) => acf[k] >= 0.9 * best));
}
if ($("w-anatomy")) {
  const N = 480, H = 48;
  const draw = () => {
    const trend = +ctrl("a-trend").value, amp = +ctrl("a-amp").value, period = +ctrl("a-period").value, noise = +ctrl("a-noise").value;
    const shift = ctrl("a-shift").checked, origin = +ctrl("a-origin").value;
    const g = gaussian(mulberry32(11));
    const x = Array.from({ length: N }, (_, t) => 50 + trend * (t - N / 2) + amp * Math.sin((2 * Math.PI * t) / period) + (shift && t >= 300 ? 15 : 0) + noise * g());
    const past = x.slice(0, origin);
    const acf = acfOf(past, Math.min(150, Math.floor(past.length / 3)));
    const m = seasonFromAcf(acf);
    const steps = Array.from({ length: H }, (_, i) => i + 1);
    const snaive = steps.map((s) => past[past.length - 1 + s - m * Math.ceil(s / m)]);
    const last = steps.map(() => past[past.length - 1]);
    const truth = x.slice(origin, origin + H);
    const mae = (f: number[]) => f.reduce((s, v, i) => s + Math.abs(v - truth[i]), 0) / H;
    const W = 720, Hp = 250, ml = 34;
    const svg = svgOf("w-anatomy", W, Hp)!;
    svg.selectAll("*").remove();
    const xs = scaleLinear().domain([0, N]).range([ml, W - 8]);
    const lo = Math.min(...x) - 5, hi = Math.max(...x) + 5;
    const ys = scaleLinear().domain([lo, hi]).range([Hp - 22, 16]);
    svg.append("rect").attr("x", xs(origin - 256)).attr("width", xs(origin) - xs(origin - 256)).attr("y", 10).attr("height", Hp - 32).attr("fill", cssVar("--accent-wash"));
    grid(svg, ys, ys.ticks(4), ml, W - 8);
    const ln = line<number>().x((_, i) => xs(i)).y((d) => ys(d));
    svg.append("path").attr("d", ln(x.slice(0, origin))).attr("fill", "none").attr("stroke", cssVar("--ink-2")).attr("stroke-width", 1.3);
    svg.append("path").attr("d", line<number>().x((_, i) => xs(origin + i)).y((d) => ys(d))(x.slice(origin))).attr("fill", "none").attr("stroke", cssVar("--ink")).attr("stroke-dasharray", "3 2").attr("opacity", 0.6);
    const lf = line<number>().x((_, i) => xs(origin + i)).y((d) => ys(d));
    svg.append("path").attr("d", lf(snaive)).attr("fill", "none").attr("stroke", cssVar("--student")).attr("stroke-width", 2.2);
    svg.append("path").attr("d", lf(last)).attr("fill", "none").attr("stroke", cssVar("--teacher")).attr("stroke-width", 1.8).attr("stroke-dasharray", "6 3");
    svg.append("line").attr("x1", xs(origin)).attr("x2", xs(origin)).attr("y1", 10).attr("y2", Hp - 22).attr("stroke", cssVar("--ink"));
    const lab = (xp: number, t: string, anchor = "middle") => svg.append("text").attr("x", xp).attr("y", Hp - 6).attr("text-anchor", anchor).style("fill", cssVar("--ink-2")).text(t);
    lab((xs(origin - 256) + xs(origin)) / 2, "context (what the model sees)");
    lab(xs(origin), "origin");
    lab(xs(origin + H / 2) + 4, "horizon", "start");
    stats("w-anatomy-stats", [["season detected by ACF", m === 1 ? "none (m = 1)" : `${m} steps`], ["MAE seasonal naive", mae(snaive).toFixed(2)], ["MAE last value", mae(last).toFixed(2)]]);
    // ACF plot
    const sa = svgOf("w-acf", W, 150)!;
    sa.selectAll("*").remove();
    const xa = scaleLinear().domain([0, acf.length]).range([ml, W - 8]);
    const ya = scaleLinear().domain([-1, 1]).range([140, 8]);
    grid(sa, ya, [-1, 0, 1], ml, W - 8);
    acf.forEach((v, k) => sa.append("line").attr("x1", xa(k)).attr("x2", xa(k)).attr("y1", ya(0)).attr("y2", ya(v)).attr("stroke", k === m ? cssVar("--student") : cssVar("--axis")).attr("stroke-width", k === m ? 3 : 1.5));
    sa.append("line").attr("x1", ml).attr("x2", W - 8).attr("y1", ya(0.3)).attr("y2", ya(0.3)).attr("stroke", cssVar("--teacher")).attr("stroke-dasharray", "4 3");
    sa.append("text").attr("x", W - 10).attr("y", ya(0.3) - 4).attr("text-anchor", "end").text("peak threshold 0.3");
    sa.append("text").attr("x", W - 10).attr("y", 146).attr("text-anchor", "end").text("lag k →");
  };
  for (const id of ["a-trend", "a-amp", "a-period", "a-noise", "a-shift", "a-origin"]) $(id)!.addEventListener("input", draw);
  draw();
}

// ------------------------------------------------------------------ ch2 + ch4: pinball loss minimiser
function mountPinball(prefix: string, allowTeacher: boolean) {
  const rnd = gaussian(mulberry32(7));
  const samples = Array.from({ length: 300 }, () => 10 * Math.exp(0.5 * rnd()));
  // teacher: 13 quantiles of a lognormal(log 11, 0.35)
  const zs = [-2.3263, -1.6449, -1.2816, -0.8416, -0.5244, -0.2533, 0, 0.2533, 0.5244, 0.8416, 1.2816, 1.6449, 2.3263];
  const tl = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99];
  const tv = zs.map((z) => 11 * Math.exp(0.35 * z));
  const edges = [0, ...tl.slice(1).map((v, i) => (v + tl[i]) / 2), 1];
  const tw = tl.map((_, i) => edges[i + 1] - edges[i]);
  const draw = () => {
    const tau = +ctrl(`${prefix}-tau`).value;
    $(`${prefix}-tauv`)!.textContent = tau.toFixed(2);
    const teacher = allowTeacher && (ctrl(`${prefix}-src`) as unknown as HTMLSelectElement).value === "teacher";
    const obs = teacher ? tv : samples, w = teacher ? tw : samples.map(() => 1 / samples.length);
    const g = Array.from({ length: 301 }, (_, i) => 2 + (i * 28) / 300);
    const L = g.map((q) => obs.reduce((s, y, k) => s + w[k] * (y >= q ? tau * (y - q) : (1 - tau) * (q - y)), 0));
    const iMin = L.indexOf(Math.min(...L));
    const ord = obs.map((y, k) => [y, w[k]] as const).sort((a, b) => a[0] - b[0]);
    let acc = 0, qTau = ord[ord.length - 1][0];
    for (const [y, wk] of ord) { acc += wk; if (acc >= tau - 1e-12) { qTau = y; break; } }
    const W = 700, H = 240, ml = 44;
    const svg = svgOf(prefix, W, H)!;
    svg.selectAll("*").remove();
    const xs = scaleLinear().domain([2, 30]).range([ml, W - 10]);
    const ys = scaleLinear().domain([0, Math.max(...L)]).nice().range([H - 34, 12]);
    grid(svg, ys, ys.ticks(4), ml, W - 10, (v) => v.toFixed(1));
    for (const t of xs.ticks(7)) svg.append("text").attr("x", xs(t)).attr("y", H - 20).attr("text-anchor", "middle").text(t);
    svg.append("text").attr("x", W - 10).attr("y", H - 4).attr("text-anchor", "end").text("candidate forecast q →");
    svg.append("text").attr("x", ml + 4).attr("y", 20).text("expected pinball loss");
    obs.forEach((v, k) => svg.append("line").attr("x1", xs(v)).attr("x2", xs(v)).attr("y1", H - 34).attr("y2", H - 34 - (teacher ? 6 + 120 * w[k] : 6))
      .attr("stroke", cssVar("--teacher")).attr("stroke-width", teacher ? 2 : 1).attr("opacity", teacher ? 0.9 : 0.35));
    svg.append("path").attr("d", line<number>().x((_, i) => xs(g[i])).y((d) => ys(d))(L)).attr("fill", "none").attr("stroke", cssVar("--student")).attr("stroke-width", 2);
    svg.append("line").attr("x1", xs(qTau)).attr("x2", xs(qTau)).attr("y1", 12).attr("y2", H - 34).attr("stroke", cssVar("--ink-2")).attr("stroke-dasharray", "4 3");
    svg.append("circle").attr("cx", xs(g[iMin])).attr("cy", ys(L[iMin])).attr("r", 5).attr("fill", cssVar("--student")).attr("stroke", cssVar("--surface")).attr("stroke-width", 2);
    stats(`${prefix}-stats`, [["minimum of the loss at q =", g[iMin].toFixed(2)], [`${tau.toFixed(2)}-quantile of the ${teacher ? "teacher's distribution" : "observations"}`, qTau.toFixed(2)]]);
  };
  $(`${prefix}-tau`)!.addEventListener("input", draw);
  if (allowTeacher) $(`${prefix}-src`)!.addEventListener("change", draw);
  draw();
}
if ($("w-pinball")) mountPinball("w-pinball", false);
if ($("w-kdpin")) mountPinball("w-kdpin", true);

// ------------------------------------------------------------------ ch2: calibration vs sharpness
if ($("w-calib")) {
  // truth Y ~ N(0,1); forecast N(mu, s). E[CRPS] = E|X - Y| - E|X - X'|/2, X - Y ~ N(mu, s^2 + 1).
  const eAbs = (mu: number, sd: number) => sd * Math.sqrt(2 / Math.PI) * Math.exp(-(mu * mu) / (2 * sd * sd)) + mu * (1 - 2 * Phi(-mu / sd));
  const crps = (mu: number, s: number) => eAbs(mu, Math.sqrt(s * s + 1)) - s / Math.sqrt(Math.PI);
  const draw = () => {
    const mu = +ctrl("c-bias").value, s = +ctrl("c-spread").value;
    $("c-biasv")!.textContent = mu.toFixed(2); $("c-spreadv")!.textContent = s.toFixed(2);
    const lo = mu - 1.2816 * s, hi = mu + 1.2816 * s;
    const cov = Phi(hi) - Phi(lo);
    const W = 700, H = 220, ml = 20;
    const svg = svgOf("w-calib", W, H)!;
    svg.selectAll("*").remove();
    const xs = scaleLinear().domain([-5, 5]).range([ml, W / 2 + 60]);
    const xv = Array.from({ length: 201 }, (_, i) => -5 + i * 0.05);
    const ys = scaleLinear().domain([0, Math.max(0.42, 0.42 / s)]).range([H - 26, 12]);
    svg.append("path").attr("d", area<number>().x((v) => xs(v)).y0(ys(0)).y1((v) => ys(phi((v - mu) / s) / s))(xv.filter((v) => v >= lo && v <= hi))).attr("fill", cssVar("--student")).attr("opacity", 0.2);
    svg.append("path").attr("d", line<number>().x((v) => xs(v)).y((v) => ys(phi(v)))(xv)).attr("fill", "none").attr("stroke", cssVar("--ink")).attr("stroke-width", 1.5).attr("stroke-dasharray", "4 3");
    svg.append("path").attr("d", line<number>().x((v) => xs(v)).y((v) => ys(phi((v - mu) / s) / s))(xv)).attr("fill", "none").attr("stroke", cssVar("--student")).attr("stroke-width", 2);
    svg.append("text").attr("x", ml).attr("y", H - 8).text("dashed: what really happens N(0,1) · blue: your forecast, shaded = its 80% interval");
    // right: E[CRPS] vs spread
    const x2 = scaleLinear().domain([0.2, 3]).range([W / 2 + 110, W - 10]);
    const sv = Array.from({ length: 140 }, (_, i) => 0.2 + i * 0.02);
    const cv = sv.map((v) => crps(mu, v));
    const y2 = scaleLinear().domain([0.5, 1.6]).range([H - 26, 12]);
    grid(svg, y2, [0.6, 1.0, 1.4], W / 2 + 110, W - 10, (v) => v.toFixed(1));
    svg.append("path").attr("d", line<number>().x((_, i) => x2(sv[i])).y((d) => y2(Math.min(1.6, d)))(cv)).attr("fill", "none").attr("stroke", cssVar("--student")).attr("stroke-width", 2);
    svg.append("circle").attr("cx", x2(s)).attr("cy", y2(Math.min(1.6, crps(mu, s)))).attr("r", 5).attr("fill", cssVar("--student"));
    svg.append("text").attr("x", W - 10).attr("y", H - 8).attr("text-anchor", "end").text("forecast spread →   (curve: expected CRPS)");
    stats("w-calib-stats", [["80% interval contains the truth", `${(100 * cov).toFixed(0)}% of the time`], ["interval width (sharpness)", (hi - lo).toFixed(2)], ["expected CRPS", crps(mu, s).toFixed(3)], ["best possible", (1 / Math.sqrt(Math.PI)).toFixed(3)]]);
  };
  for (const id of ["c-bias", "c-spread"]) $(id)!.addEventListener("input", draw);
  draw();
}

// ------------------------------------------------------------------ data-driven widgets
const needResults = $("w-bias") || $("w-latency") || $("w-ladder");
const needGallery = $("w-scaler") || $("w-tokens");
const [res, gal] = await Promise.all([
  needResults ? loadJSON<Results>("results.json") : Promise.resolve(null),
  needGallery ? loadJSON<Gallery>("gallery.json") : Promise.resolve(null),
]);

if (res && $("w-bias")) {
  const b = res.crps_bias;
  const pct = (v: number) => `${v > b.exact ? "+" : ""}${(100 * (v / b.exact - 1)).toFixed(1)}%`;
  $("w-bias")!.innerHTML = `<table><thead><tr><th>quantile levels used</th><th class="num">equal weights</th><th class="num">bias</th><th class="num">probability-mass weights</th><th class="num">bias</th></tr></thead><tbody>` +
    Object.keys(b.equal_weights).map((k) => `<tr><td>${k}</td><td class="num">${b.equal_weights[k].toFixed(3)}</td><td class="num">${pct(b.equal_weights[k])}</td><td class="num">${b.mass_weights[k].toFixed(3)}</td><td class="num">${pct(b.mass_weights[k])}</td></tr>`).join("") +
    `</tbody></table><p class="small muted" style="margin:8px 0 0">Expected score of a perfect N(0,1) forecast; exact CRPS = 1/√π = ${b.exact.toFixed(3)}. Computed analytically by <code>scripts/build_web_data.py</code>.</p>`;
}

if (res && $("w-ladder")) {
  const rows = res.suites.fresh_fr_power.rows.filter((r) => !r.model.endsWith("onnx-fp32"));
  $("w-ladder")!.innerHTML = `<table><thead><tr><th>model</th><th class="num">rel. WQL</th><th class="num">95% CI</th><th class="num">80% coverage</th></tr></thead><tbody>${rows.map((r) =>
    `<tr><td>${r.model}</td><td class="num">${r.rel_wql.toFixed(3)}</td><td class="num">[${r.rel_wql_ci[0].toFixed(3)}, ${r.rel_wql_ci[1].toFixed(3)}]</td><td class="num">${(100 * r.mean_cov80).toFixed(0)}%</td></tr>`).join("")}</tbody></table>
    <p class="small muted" style="margin:8px 0 0">${rows[0].n_windows} windows from ${rows[0].n_clusters} series; CIs from a cluster bootstrap over series.</p>`;
}

if (res && $("w-latency")) {
  const rows = res.suites.shapes.rows.filter((r) => r.latency_ms_t1_b1 && !r.model.endsWith("onnx-fp32")).sort((a, b) => a.n_params - b.n_params);
  $("w-latency")!.innerHTML = `<table><thead><tr><th>model</th><th class="num">parameters</th><th class="num">1 thread, batch 1</th><th class="num">8 threads, batch 32</th><th class="num">rel. WQL (shapes)</th></tr></thead><tbody>${rows.map((r) =>
    `<tr><td>${r.model}</td><td class="num">${r.n_params ? (r.n_params / 1e6).toFixed(2) + "M" : "–"}</td><td class="num">${r.latency_ms_t1_b1!.toFixed(1)} ms</td><td class="num">${r.latency_ms_t8_b32 ? r.latency_ms_t8_b32.toFixed(0) + " ms" : "–"}</td><td class="num">${r.rel_wql.toFixed(3)}</td></tr>`).join("")}</tbody></table>
    <p class="small muted" style="margin:8px 0 0">CPU (Python, PyTorch or onnxruntime), p50 after warm-up. Same machine for every model.</p>`;
}

// ch3: robust scaling
if (gal && $("w-scaler")) {
  const sh = gal.shapes.find((s) => s.name === "spikes")!;
  const x = sh.context.filter((v): v is number => v !== null);
  const mean = x.reduce((a, b) => a + b, 0) / x.length;
  const sd = Math.sqrt(x.reduce((a, b) => a + (b - mean) ** 2, 0) / x.length);
  const sorted = [...x].sort((a, b) => a - b), med = sorted[Math.floor((x.length - 1) / 2)];
  const mad = x.reduce((a, b) => a + Math.abs(b - med), 0) / x.length;
  const panels: [string, number[]][] = [["raw values", x], ["(x − mean) / std", x.map((v) => (v - mean) / sd)], ["asinh((x − median) / MAD)  ← ours", x.map((v) => Math.asinh((v - med) / mad))]];
  const W = 720, H = 150, pw = W / 3;
  const svg = svgOf("w-scaler", W, H)!;
  panels.forEach(([title, z], k) => {
    const lo = Math.min(...z), hi = Math.max(...z);
    const xs = scaleLinear().domain([0, z.length]).range([k * pw + 8, (k + 1) * pw - 8]);
    const ys = scaleLinear().domain([lo, hi]).range([H - 10, 22]);
    svg.append("path").attr("d", line<number>().x((_, i) => xs(i)).y((d) => ys(d))(z)).attr("fill", "none").attr("stroke", k ? cssVar("--student") : cssVar("--ink-2")).attr("stroke-width", 1);
    svg.append("text").attr("x", k * pw + 8).attr("y", 12).style("fill", cssVar("--ink-2")).text(title);
  });
  const amp = (z: number[]) => { const s = [...z].sort((a, b) => a - b); return s[Math.floor(0.9 * s.length)] - s[Math.floor(0.1 * s.length)]; };
  stats("w-scaler-stats", [["std (pulled up by spikes)", sd.toFixed(2)], ["MAD", mad.toFixed(2)], ["daily wave height after mean/std", amp(panels[1][1]).toFixed(2)], ["after median/MAD + asinh", amp(panels[2][1]).toFixed(2)]]);
}

// ch3: tokenisation (patches vs bins)
if (gal && $("w-tokens")) {
  const sh = gal.shapes.find((s) => s.name === "two_seasonalities")!;
  const x = sh.context.filter((v): v is number => v !== null).slice(-512);
  const draw = () => {
    const p = [4, 8, 16, 32, 64][+ctrl("t-patch").value];
    const bins = [8, 16, 32, 64, 256, 4096][+ctrl("t-bins").value];
    $("t-patchv")!.textContent = String(p); $("t-binsv")!.textContent = String(bins);
    const meanAbs = x.reduce((a, b) => a + Math.abs(b), 0) / x.length;
    const z = x.map((v) => v / meanAbs); // Chronos-1 style mean scaling
    const lo = -15, hi = 15, width = (hi - lo) / bins;
    const zq = z.map((v) => lo + (Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / width))) + 0.5) * width);
    const W = 720, H = 220, ml = 8;
    const svg = svgOf("w-tokens", W, H)!;
    svg.selectAll("*").remove();
    const show = 160, xs = scaleLinear().domain([0, show]).range([ml, W - 8]);
    const seg = z.slice(-show), segq = zq.slice(-show);
    const ys = scaleLinear().domain([Math.min(...seg) - 0.05, Math.max(...seg) + 0.05]).range([H - 12, 12]);
    for (let s = 0; s < show; s += p) if ((s / p) % 2 === 0) svg.append("rect").attr("x", xs(s)).attr("width", xs(Math.min(s + p, show)) - xs(s)).attr("y", 8).attr("height", H - 16).attr("fill", cssVar("--accent-wash"));
    svg.append("path").attr("d", line<number>().x((_, i) => xs(i)).y((d) => ys(d))(seg)).attr("fill", "none").attr("stroke", cssVar("--ink-2")).attr("stroke-width", 1.4);
    svg.append("path").attr("d", line<number>().x((_, i) => xs(i)).y((d) => ys(d))(segq)).attr("fill", "none").attr("stroke", cssVar("--teacher")).attr("stroke-width", 1.4);
    const n = 512 / p, err = Math.sqrt(z.reduce((s, v, i) => s + (v - zq[i]) ** 2, 0) / z.length);
    stats("w-tokens-stats", [["tokens for 512 steps", String(n)], ["attention pairs (n²)", (n * n).toLocaleString()], ["vs one token per step", `${((512 * 512) / (n * n)).toFixed(0)}× cheaper`], [`bins: rounding error (in units of mean |x|)`, err.toFixed(4)]]);
  };
  for (const id of ["t-patch", "t-bins"]) $(id)!.addEventListener("input", draw);
  draw();
}

// ch3: monotone head
if ($("w-head")) {
  const sp = (v: number) => Math.log1p(Math.exp(v));
  const draw = () => {
    const a = [1, 2, 3, 4].map((k) => +ctrl(`h-${k}`).value);
    const free = [a[0], a[1], 0, a[2], a[3]];
    const q3 = -sp(a[1]), q1 = q3 - sp(a[0]), q7 = sp(a[2]), q9 = q7 + sp(a[3]);
    const mono = [q1, q3, 0, q7, q9];
    const labels = ["0.1", "0.3", "0.5", "0.7", "0.9"];
    const W = 720, H = 150;
    const svg = svgOf("w-head", W, H)!;
    svg.selectAll("*").remove();
    const xs = scaleLinear().domain([-6, 6]).range([150, W - 20]);
    const row = (vals: number[], y: number, title: string) => {
      const crossed = vals.some((v, i) => i > 0 && v < vals[i - 1]);
      svg.append("text").attr("x", 8).attr("y", y + 4).style("fill", cssVar("--ink-2")).text(title);
      svg.append("line").attr("x1", xs(-6)).attr("x2", xs(6)).attr("y1", y).attr("y2", y).attr("stroke", cssVar("--axis"));
      vals.forEach((v, i) => {
        svg.append("circle").attr("cx", xs(v)).attr("cy", y).attr("r", 5).attr("fill", crossed ? cssVar("--teacher") : cssVar("--student"));
        svg.append("text").attr("x", xs(v)).attr("y", y - 10 - (i % 2) * 11).attr("text-anchor", "middle").text(labels[i]);
      });
      svg.append("text").attr("x", W - 20).attr("y", y + 20).attr("text-anchor", "end").style("fill", crossed ? cssVar("--teacher") : cssVar("--muted")).text(crossed ? "quantiles cross: invalid distribution" : "ordered");
    };
    row(free, 48, "raw outputs as quantiles");
    row(mono, 118, "median + softplus gaps");
  };
  for (const k of [1, 2, 3, 4]) $(`h-${k}`)!.addEventListener("input", draw);
  draw();
}

// ch4: teachers on different input families
if ($("w-families")) {
  interface Fam { teachers: Record<string, Record<string, { movement_median: number; diversity: number; width80_median: number }>>; examples: Record<string, { context: number[]; q: number[][] }[]>; example_levels: number[]; n_per_family: number }
  loadJSON<Fam>("input_families.json").then((f) => {
    const fams = Object.keys(f.examples);
    const box = $("w-families")!;
    box.innerHTML = `<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px">${fams.map((k) => `<div><div class="small muted">${k.replace("_", " ")}</div><svg data-fam="${k}"></svg></div>`).join("")}</div>`;
    for (const k of fams) {
      const ex = f.examples[k][0];
      drawFan(box.querySelector(`svg[data-fam="${k}"]`) as SVGSVGElement, { context: ex.context, q: ex.q, levels: f.example_levels, color: cssVar("--teacher"), width: 160, height: 80, showContext: 128 });
    }
    const t = Object.keys(f.teachers);
    $("w-families-table")!.innerHTML = `<table><thead><tr><th>contexts</th>${t.map((n) => `<th class="num">movement · ${n}</th>`).join("")}${t.map((n) => `<th class="num">diversity · ${n}</th>`).join("")}</tr></thead><tbody>${fams.map((k) =>
      `<tr><td>${k.replace("_", " ")}</td>${t.map((n) => `<td class="num">${f.teachers[n][k].movement_median.toFixed(3)}</td>`).join("")}${t.map((n) => `<td class="num">${f.teachers[n][k].diversity.toFixed(3)}</td>`).join("")}</tr>`).join("")}</tbody></table>
      <p class="small muted" style="margin:8px 0 0">${f.n_per_family} contexts per family, measured in each context's own scale. <b>movement</b> = how much the teacher's median moves over the horizon, relative to its 80% band. <b>diversity</b> = how different the answers to different contexts are. Source: <code>scripts/input_families.py</code> → <code>results/input_families.json</code>.</p>`;
  });
}

// ch4: KernelSynth sampler (Gaussian process with composed kernels, in the browser)
if ($("w-ks")) {
  const n = 200, t = Array.from({ length: n }, (_, i) => i / (n - 1));
  let seed = 1;
  const kern = (kind: string, a: number, b: number) => {
    const d = a - b, period = +ctrl("k-period").value / n, ell = +ctrl("k-ell").value;
    if (kind === "periodic") return Math.exp((-2 * Math.sin((Math.PI * Math.abs(d)) / period) ** 2) / 0.8 ** 2);
    if (kind === "rbf") return Math.exp(-0.5 * (d / ell) ** 2);
    if (kind === "linear") return (a - 0.5) * (b - 0.5) * 8;
    return a === b ? 0.15 : 0; // white
  };
  const draw = () => {
    const on = ["periodic", "rbf", "linear", "white"].filter((k) => ctrl(`k-${k}`).checked);
    const op = (ctrl("k-op") as unknown as HTMLSelectElement).value;
    $("k-periodv")!.textContent = ctrl("k-period").value; $("k-ellv")!.textContent = (+ctrl("k-ell").value).toFixed(2);
    const K = t.map((a) => t.map((b) => {
      if (!on.length) return a === b ? 1 : 0;
      return on.map((k) => kern(k, a, b)).reduce((x, y) => (op === "+" ? x + y : x * y));
    }));
    for (let i = 0; i < n; i++) K[i][i] += 1e-4;
    // Cholesky
    const L = t.map(() => new Array(n).fill(0));
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let s = K[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-10)) : s / L[j][j];
    }
    const g = gaussian(mulberry32(seed));
    const samples = [0, 1, 2].map(() => { const z = t.map(() => g()); return t.map((_, i) => { let s = 0; for (let k = 0; k <= i; k++) s += L[i][k] * z[k]; return s; }); });
    const W = 720, H = 200;
    const svg = svgOf("w-ks", W, H)!;
    svg.selectAll("*").remove();
    const all = samples.flat(), xs = scaleLinear().domain([0, n - 1]).range([8, W - 8]);
    const ys = scaleLinear().domain([Math.min(...all), Math.max(...all)]).range([H - 8, 8]);
    const cols = [cssVar("--student"), cssVar("--teacher"), cssVar("--comparator")];
    samples.forEach((s, k) => svg.append("path").attr("d", line<number>().x((_, i) => xs(i)).y((d) => ys(d))(s)).attr("fill", "none").attr("stroke", cols[k]).attr("stroke-width", 1.5));
    $("k-formula")!.textContent = on.length ? `k(t, t') = ${on.join(` ${op === "+" ? "+" : "×"} `)}` : "no kernel: plain white noise";
  };
  for (const id of ["k-periodic", "k-rbf", "k-linear", "k-white", "k-op", "k-period", "k-ell"]) $(id)!.addEventListener("input", draw);
  $("k-resample")!.addEventListener("click", () => { seed += 1; draw(); });
  draw();
}

// ch5: int8 quantisation of a weight matrix
if ($("w-quant")) {
  const g = gaussian(mulberry32(3));
  const base = Array.from({ length: 4 }, () => Array.from({ length: 24 }, () => 0.05 * g()));
  const draw = () => {
    const outlier = +ctrl("q-outlier").value, bits = +ctrl("q-bits").value, perCh = ctrl("q-perch").checked;
    $("q-outlierv")!.textContent = `${outlier}×`; $("q-bitsv")!.textContent = String(bits);
    const w = base.map((r) => [...r]);
    w[0][5] = 0.05 * outlier; // one large weight in channel 0
    const qmax = 2 ** (bits - 1) - 1;
    const allMax = Math.max(...w.flat().map(Math.abs));
    const scales = w.map((r) => (perCh ? Math.max(...r.map(Math.abs)) : allMax) / qmax);
    const wq = w.map((r, c) => r.map((v) => Math.round(v / scales[c]) * scales[c]));
    const W = 720, H = 230, rowH = (H - 20) / 4;
    const svg = svgOf("w-quant", W, H)!;
    svg.selectAll("*").remove();
    const xs = scaleLinear().domain([-0.17, 0.17]).range([110, W - 10]);
    let se = 0, sv = 0;
    w.forEach((r, c) => {
      const y0 = 10 + c * rowH + rowH / 2;
      svg.append("text").attr("x", 8).attr("y", y0 + 4).text(`channel ${c}${c === 0 ? " (has outlier)" : ""}`);
      for (let k = -qmax; k <= qmax; k++) { const v = k * scales[c]; if (Math.abs(v) <= 0.17) svg.append("line").attr("x1", xs(v)).attr("x2", xs(v)).attr("y1", y0 - 10).attr("y2", y0 + 10).attr("stroke", cssVar("--grid")); }
      r.forEach((v, i) => {
        se += (v - wq[c][i]) ** 2; sv += v * v;
        const xv = xs(Math.max(-0.17, Math.min(0.17, v)));
        svg.append("line").attr("x1", xv).attr("x2", xs(Math.max(-0.17, Math.min(0.17, wq[c][i])))).attr("y1", y0).attr("y2", y0).attr("stroke", cssVar("--teacher")).attr("stroke-width", 2);
        svg.append("circle").attr("cx", xv).attr("cy", y0).attr("r", 3).attr("fill", cssVar("--student"));
      });
    });
    const used = new Set(wq.flat().map((v, i) => Math.round(v / scales[Math.floor(i / 24)]))).size;
    stats("w-quant-stats", [["grid lines = representable values", `${2 * qmax + 1} per channel`], ["distinct values actually used", String(used)], ["relative error ‖w − q(w)‖ / ‖w‖", `${(100 * Math.sqrt(se / sv)).toFixed(1)}%`]]);
  };
  for (const id of ["q-outlier", "q-bits", "q-perch"]) $(id)!.addEventListener("input", draw);
  draw();
}

// ch4: teacher rollouts
if ($("w-rollout")) {
  interface Roll { stats: Record<string, Record<string, number[]>>; examples: Record<string, number[][]>; blocks: number }
  loadJSON<Roll>("rollouts.json").then((r) => {
    const teachers = [...new Set(Object.keys(r.stats).map((k) => k.split("|")[0]))];
    const sel = $("r-teacher") as HTMLSelectElement;
    for (const t of teachers) sel.add(new Option(t, t));
    const strategies = ["median", "independent", "copula"];
    const names: Record<string, string> = { median: "median", independent: "independent", copula: "correlated" };
    const cols = () => [cssVar("--teacher"), cssVar("--baseline"), cssVar("--student")];
    const draw = () => {
      const src = (ctrl("r-src") as unknown as HTMLSelectElement).value, metric = (ctrl("r-metric") as unknown as HTMLSelectElement).value;
      const W = 720, H = 220, ml = 40;
      const svg = svgOf("w-rollout", W, H)!;
      svg.selectAll("*").remove();
      const series = strategies.map((s) => r.stats[`${sel.value}|${src}|${s}`][metric]);
      const all = series.flat();
      const xs = scaleLinear().domain([1, r.blocks]).range([ml, W - 110]);
      const ys = scaleLinear().domain([0, Math.max(...all, metric === "drift" ? 1 : 1.2)]).nice().range([H - 24, 10]);
      grid(svg, ys, ys.ticks(5), ml, W - 110, (v) => v.toFixed(1));
      if (metric !== "drift") svg.append("line").attr("x1", ml).attr("x2", W - 110).attr("y1", ys(1)).attr("y2", ys(1)).attr("stroke", cssVar("--ink")).attr("stroke-dasharray", "4 3");
      series.forEach((v, k) => {
        svg.append("path").attr("d", line<number>().x((_, i) => xs(i + 1)).y((d) => ys(d))(v)).attr("fill", "none").attr("stroke", cols()[k]).attr("stroke-width", 2);
        svg.append("text").attr("x", W - 104).attr("y", ys(v[v.length - 1]) + 4).style("fill", cssVar("--ink-2")).text(names[strategies[k]]);
      });
      for (const t of [1, 4, 8, 12, 16]) svg.append("text").attr("x", xs(t)).attr("y", H - 8).attr("text-anchor", "middle").text(`block ${t}`);
      const ex = $("w-rollout-ex")!;
      ex.innerHTML = strategies.map((s) => `<div><div class="small muted">${names[s]}</div><svg data-s="${s}"></svg></div>`).join("");
      strategies.forEach((s, k) => {
        const v = r.examples[`${src}|${s}`]?.[0];
        if (!v) return;
        const real = 128; // examples start 256 steps into the 512-step real context, every 2nd step
        const e = select(ex.querySelector(`svg[data-s="${s}"]`) as SVGSVGElement).attr("viewBox", "0 0 220 80").attr("width", "100%");
        const fin = v.filter(Number.isFinite), lo = Math.min(...fin), hi = Math.max(...fin);
        const xe = scaleLinear().domain([0, v.length]).range([2, 218]), ye = scaleLinear().domain([lo, hi]).range([76, 4]);
        e.append("path").attr("d", line<number>().x((_, i) => xe(i)).y((d) => ye(d))(v.slice(0, real))).attr("fill", "none").attr("stroke", cssVar("--ink-2")).attr("stroke-width", 1);
        e.append("path").attr("d", line<number>().x((_, i) => xe(real + i)).y((d) => ye(d))(v.slice(real))).attr("fill", "none").attr("stroke", cols()[k]).attr("stroke-width", 1);
      });
    };
    for (const id of ["r-teacher", "r-src", "r-metric"]) $(id)!.addEventListener("input", draw);
    draw();
  });
}

// ch6: ablation study (Stage A) -------------------------------------------------------------------------------
if ($("w-sa") || document.querySelector("[data-pair],[data-arm],[data-zoo]") || $("w-stages")) {
  interface SRow { rel_wql: number; rel_wql_ci: [number, number] }
  interface SContrast { a: string; b: string; question: string; ratio: [number, number, number] }
  interface SSuite { arms: Record<string, SRow>; pairs?: SContrast[]; zero_shot_models: Record<string, SRow> }
  type Study = Record<string, { suites: Record<string, SSuite> }>;
  loadJSON<Study>("study.json").then((st) => {
    const fmt = (r: [number, number, number]) => `${r[0].toFixed(2)} [${r[1].toFixed(2)}, ${r[2].toFixed(2)}]`;
    for (const el of document.querySelectorAll<HTMLElement>("[data-pair]")) {
      const [stage, suite, a, b] = el.dataset.pair!.split("|");
      const c = st[stage]?.suites[suite]?.pairs?.find((p) => p.a === a && p.b === b);
      el.textContent = c ? fmt(c.ratio) : "(not computed yet)";
    }
    for (const el of document.querySelectorAll<HTMLElement>("[data-arm]")) {
      const [stage, suite, arm] = el.dataset.arm!.split("|");
      const r = st[stage]?.suites[suite]?.arms[arm];
      el.textContent = r ? r.rel_wql.toFixed(2) : "(n/a)";
    }
    for (const el of document.querySelectorAll<HTMLElement>("[data-zoo]")) {
      const [stage, suite, m] = el.dataset.zoo!.split("|");
      const r = st[stage]?.suites[suite]?.zero_shot_models[m];
      el.textContent = r ? r.rel_wql.toFixed(2) : "(n/a)";
    }
    if ($("w-stages")) {
      const rows: [string, string, string][] = [
        ["stage_a", "A", "Which inputs should the teacher label?"],
        ["stage_d_pilot", "D (pilot)", "Does light fine-tuning on your data close the gap to the big models?"],
        ["stage_c_lite", "C (lite)", "Do a bigger student and more, mixed data close the zero-shot gap?"],
        ["stage_b", "B", "Which teacher (or combination) and how much teacher vs truth (α)? 3 seeds"],
      ];
      $("w-stages")!.innerHTML = rows.map(([k, n, q]) => `<tr><td><b>${n}</b></td><td>${q}</td><td>${st[k] ? `<a href="study.html#${k}">results</a>` : '<span class="muted">queued / running</span>'}</td></tr>`).join("");
    }
    if ($("w-sa")) {
      const draw = () => {
        const suite = (ctrl("sa-suite") as unknown as HTMLSelectElement).value;
        const pairs = st.stage_a?.suites[suite]?.pairs ?? [];
        const W = 720, rowH = 34, H = pairs.length * rowH + 30, ml = 330;
        const svg = svgOf("w-sa", W, H)!;
        svg.selectAll("*").remove();
        const x = scaleLinear().domain([Math.log(0.25), Math.log(6)]).range([ml, W - 50]);
        const lx = (v: number) => x(Math.log(Math.min(6, Math.max(0.25, v))));
        svg.append("line").attr("x1", lx(1)).attr("x2", lx(1)).attr("y1", 4).attr("y2", H - 22).attr("stroke", cssVar("--ink")).attr("stroke-dasharray", "3 3");
        for (const t of [0.25, 0.5, 1, 2, 4]) svg.append("text").attr("x", lx(t)).attr("y", H - 6).attr("text-anchor", "middle").text(t);
        pairs.forEach((p, i) => {
          const y = 18 + i * rowH, [r, lo, hi] = p.ratio;
          const col = hi < 1 ? cssVar("--student") : lo > 1 ? cssVar("--teacher") : cssVar("--baseline");
          svg.append("text").attr("x", ml - 10).attr("y", y - 3).attr("text-anchor", "end").style("fill", cssVar("--ink")).text(p.question);
          svg.append("text").attr("x", ml - 10).attr("y", y + 11).attr("text-anchor", "end").text(`${p.a} vs ${p.b}`);
          svg.append("line").attr("x1", lx(lo)).attr("x2", lx(hi)).attr("y1", y).attr("y2", y).attr("stroke", col).attr("stroke-width", 3);
          svg.append("circle").attr("cx", lx(r)).attr("cy", y).attr("r", 5).attr("fill", col);
          svg.append("text").attr("x", lx(hi) + 6).attr("y", y + 4).text(r.toFixed(2));
        });
      };
      $("sa-suite")!.addEventListener("input", draw);
      draw();
    }
  });
}

// ch6: bias-variance of the training target -------------------------------------------------------------------
if ($("w-target")) {
  const Z90 = 1.2816;
  const ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
  const g = gaussian(mulberry32(5));
  const gtErr = ns.map((n) => {
    let s = 0;
    for (let r = 0; r < 400; r++) {
      const x = Array.from({ length: n }, () => g()).sort((a, b) => a - b);
      const pos = 0.9 * (n - 1), i = Math.floor(pos), f = pos - i;
      const q = n === 1 ? x[0] : x[i] * (1 - f) + x[Math.min(i + 1, n - 1)] * f;
      s += Math.abs(q - Z90);
    }
    return s / 400;
  });
  const draw = () => {
    const b = +ctrl("tg-bias").value, se = +ctrl("tg-spread").value;
    $("tg-biasv")!.textContent = b.toFixed(2); $("tg-spreadv")!.textContent = se.toFixed(2);
    const tErr = Math.abs(b + Z90 * se);
    const W = 720, H = 220, ml = 44;
    const svg = svgOf("w-target", W, H)!;
    svg.selectAll("*").remove();
    const x = scaleLinear().domain([0, Math.log2(1024)]).range([ml, W - 12]);
    const y = scaleLinear().domain([0, Math.max(1.5, tErr * 1.1)]).range([H - 26, 10]);
    grid(svg, y, y.ticks(4), ml, W - 12, (v) => v.toFixed(1));
    ns.forEach((n) => svg.append("text").attr("x", x(Math.log2(n))).attr("y", H - 8).attr("text-anchor", "middle").text(n));
    svg.append("text").attr("x", W - 12).attr("y", H - 30).attr("text-anchor", "end").text("observations per situation (n) →");
    svg.append("path").attr("d", line<number>().x((_, i) => x(Math.log2(ns[i]))).y((d) => y(d))(gtErr)).attr("fill", "none").attr("stroke", cssVar("--baseline")).attr("stroke-width", 2.2);
    svg.append("line").attr("x1", ml).attr("x2", W - 12).attr("y1", y(tErr)).attr("y2", y(tErr)).attr("stroke", cssVar("--teacher")).attr("stroke-width", 2.2);
    svg.append("text").attr("x", ml + 6).attr("y", 20).text("error of the learned 0.9-quantile");
    const cross = ns.find((_, i) => gtErr[i] < tErr);
    stats("w-target-stats", [["error from 1 real observation", gtErr[0].toFixed(2)], ["error from the teacher", tErr.toFixed(2)], ["ground truth wins from n =", cross ? String(cross) : "> 1024"]]);
  };
  for (const id of ["tg-bias", "tg-spread"]) $(id)!.addEventListener("input", draw);
  draw();
}

// ch6: Stage C-lite levers and comparison with the big models ---------------------------------------------------------
if ($("w-levers") || $("w-vsbig")) {
  interface SC { a: string; b: string; ratio: [number, number, number] }
  type St = Record<string, { suites: Record<string, { pairs?: SC[] }> }>;
  loadJSON<St>("study.json").then((st) => {
    const suites: [string, string][] = [["shapes", "shapes"], ["multidomain_test", "multi-domain"], ["fresh_fr_power", "electricity"]];
    const chart = (id: string, rows: [string, string][]) => {
      const c = st.stage_c_lite;
      if (!c) { $(id)!.outerHTML = `<p class="muted">Stage C-lite results not available yet.</p>`; return; }
      const W = 720, rowH = 18, groupH = suites.length * rowH + 16, H = rows.length * groupH + 26, ml = 230;
      const svg = svgOf(id, W, H)!;
      const x = scaleLinear().domain([Math.log(0.6), Math.log(1.5)]).range([ml, W - 60]);
      const lx = (v: number) => x(Math.log(Math.min(1.5, Math.max(0.6, v))));
      svg.append("line").attr("x1", lx(1)).attr("x2", lx(1)).attr("y1", 4).attr("y2", H - 20).attr("stroke", cssVar("--ink")).attr("stroke-dasharray", "3 3");
      for (const t of [0.6, 0.8, 1, 1.25, 1.5]) svg.append("text").attr("x", lx(t)).attr("y", H - 5).attr("text-anchor", "middle").text(t);
      rows.forEach(([b, label], g) => {
        const y0 = 12 + g * groupH;
        svg.append("text").attr("x", 8).attr("y", y0 + rowH).style("fill", cssVar("--ink")).text(label);
        suites.forEach(([s, sl], k) => {
          const p = c.suites[s]?.pairs?.find((q) => q.a === "kd_mix_m_300k" && q.b === b);
          const y = y0 + k * rowH + 6;
          svg.append("text").attr("x", ml - 8).attr("y", y + 4).attr("text-anchor", "end").text(sl);
          if (!p) return;
          const [r, lo, hi] = p.ratio, col = hi < 1 ? cssVar("--student") : lo > 1 ? cssVar("--teacher") : cssVar("--baseline");
          svg.append("line").attr("x1", lx(lo)).attr("x2", lx(hi)).attr("y1", y).attr("y2", y).attr("stroke", col).attr("stroke-width", 3);
          svg.append("circle").attr("cx", lx(r)).attr("cy", y).attr("r", 4.5).attr("fill", col);
          svg.append("text").attr("x", lx(hi) + 6).attr("y", y + 4).text(r.toFixed(2));
        });
      });
    };
    if ($("w-levers")) chart("w-levers", [["kd_mix_m_30k", "10× more data (300k vs 30k)"], ["gt_mix_m_300k", "teacher labels vs ground truth"], ["kd_mix_s_300k", "4× bigger student (0.95M vs 0.22M)"]]);
    if ($("w-vsbig")) chart("w-vsbig", [["zoo:toto-2.0-4m", "vs Toto 2.0 4m (4.1M)"], ["zoo:timestral-5m", "vs Timestral 5M (5.2M)"], ["zoo:timestral-209m", "vs its teacher Timestral 209M"]]);
  });
}
