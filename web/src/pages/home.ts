import { scaleLinear, scaleLog } from "d3-scale";
import { select } from "d3-selection";
import { area, line } from "d3-shape";
import { cssVar, fmtParams, hideTip, loadJSON, ratioWords, roleColor, showTip, type Manifest, type Results, type Shape, type Shapes } from "../lib/data";
import { header } from "../lib/nav";
import { CONTEXT, LEVELS, Student } from "../lib/ort";

header();
const $ = (id: string) => document.getElementById(id)!;
const L = LEVELS.map((v) => Math.round(v * 100));
const BANDS: [number, number, number][] = [[1, 99, 0.14], [10, 90, 0.26], [30, 70, 0.42]];

const [mf, res, sh] = await Promise.all([loadJSON<Manifest>("models.json"), loadJSON<Results>("results.json"), loadJSON<Shapes>("shapes.json")]);
const fresh = res.suites.fresh_fr_power;
const row = (m: string) => fresh.rows.find((r) => r.model === m)!;
const find = (title: string) => res.findings.find((f) => f.title.startsWith(title))!;
const vs4m = find("Better than Toto-2.0-4m").contrasts.find((c) => c.suite === "fresh_fr_power")!.ratio;
const vsT = find("Within").contrasts.find((c) => c.suite === "fresh_fr_power")!.ratio;
const gift = find("On GIFT-Eval").contrasts[0].ratio;
const ratio = Math.round(mf.teacher.n_params / mf.n_params);

// ---------------------------------------------------------------- claim + key numbers
$("lede").innerHTML = `We trained a ${fmtParams(mf.n_params)}-parameter forecaster to copy the probabilistic forecasts of
  ${mf.teacher.label}, a model ${ratio}× its size. On French electricity data published after both models were released, it has
  <b>${ratioWords(vs4m).text}</b> than Toto-2.0-4m and ${ratioWords(vsT).text} than its teacher. On the broader GIFT-Eval benchmark it is
  <b>${Math.round((gift[0] - 1) * 100)}% worse</b> than Toto-2.0-4m. It runs below, in your browser.`;
const numbers = (ms: string) => {
  $("numbers").innerHTML = [
    [fmtParams(mf.n_params), "parameters"],
    [`${mf.size_mb.fp32.toFixed(1)} MB`, `model download (${mf.size_mb.int8.toFixed(1)} MB in int8)`],
    [ms, "per forecast, in this browser tab"],
    [row("student").rel_wql.toFixed(3), `error on fresh electricity, relative to seasonal naive (Toto-2.0-4m: ${row("toto-2.0-4m").rel_wql.toFixed(3)})`],
  ].map(([v, l]) => `<div><b>${v}</b><span>${l}</span></div>`).join("");
};
numbers("…");

// ---------------------------------------------------------------- the model
const model = await Student.load(`${mf.onnx}.fp32.onnx`);
const lat: number[] = [];
function timed(ms: number): void {
  lat.push(ms); if (lat.length > 50) lat.shift();
  const med = [...lat].sort((a, b) => a - b)[Math.floor(lat.length / 2)];
  numbers(`${med.toFixed(1)} ms`);
  $("readout").innerHTML = `Median of the last ${lat.length} forecasts: <b>${med.toFixed(1)} ms</b> (WebAssembly, ${model.threads} thread${model.threads > 1 ? "s" : ""}).
    Reads the last ${CONTEXT} points, forecasts ${mf.horizon} steps, 13 quantiles per step.`;
}

// ---------------------------------------------------------------- draw pad
const STEPS = 640, DRAWABLE = 576, H = mf.horizon, W = 960, Hpx = 340;
const pad = $("pad") as unknown as SVGSVGElement;
const presetEl = $("preset") as HTMLSelectElement, teacherBtn = $("teacher") as HTMLButtonElement;
const x = scaleLinear().domain([0, STEPS]).range([0, W]);
const y = scaleLinear().domain([0, 100]).range([Hpx - 6, 6]);
let values: number[] = new Array(DRAWABLE).fill(NaN);
const history: number[][] = [];
let tip = -1, origin = -1, truth: number[] | null = null, teacher: number[] | null = null, showTeacher = false;
let forecast: { origin: number; q: number[][] } | null = null, busy = false, dirty = false;

function render(): void {
  const svg = select(pad).attr("viewBox", `0 0 ${W} ${Hpx}`).attr("width", "100%");
  svg.selectAll("*").remove();
  for (const t of [20, 40, 60, 80]) svg.append("line").attr("x1", 0).attr("x2", W).attr("y1", y(t)).attr("y2", y(t)).attr("stroke", cssVar("--grid"));
  svg.append("rect").attr("x", x(DRAWABLE)).attr("width", x(STEPS) - x(DRAWABLE)).attr("height", Hpx).attr("fill", cssVar("--wash"));
  const ln = line<number>().defined(Number.isFinite).x((_, i) => x(i)).y((d) => y(d));
  if (truth) svg.append("path").attr("d", line<number>().x((_, i) => x(tip + 1 + i)).y((d) => y(d))(truth))
    .attr("fill", "none").attr("stroke", cssVar("--ink")).attr("stroke-width", 1.4).attr("stroke-dasharray", "4 3");
  svg.append("path").attr("d", ln(values.map((v, i) => (i > origin ? v : NaN)))).attr("fill", "none").attr("stroke", cssVar("--ink")).attr("stroke-width", 1.4).attr("stroke-dasharray", "4 3");
  svg.append("path").attr("d", ln(values.map((v, i) => (i <= origin ? v : NaN)))).attr("fill", "none").attr("stroke", cssVar("--ink-2")).attr("stroke-width", 1.8);
  if (origin >= 0 && origin - CONTEXT + 1 > 0)
    svg.append("rect").attr("x", 0).attr("width", x(origin - CONTEXT + 1)).attr("height", Hpx).attr("fill", cssVar("--page")).attr("opacity", 0.6);
  if (forecast) {
    const s = forecast.origin + 1, c = cssVar("--student");
    for (const [a, b, op] of BANDS)
      svg.append("path").attr("d", area<number>().x((_, i) => x(s + i)).y0((d) => y(d)).y1((_, i) => y(forecast!.q[L.indexOf(b)][i]))(forecast.q[L.indexOf(a)])).attr("fill", c).attr("opacity", op);
    if (showTeacher && teacher && forecast.origin === tip)
      svg.append("path").attr("d", line<number>().x((_, i) => x(s + i)).y((d) => y(d))(teacher)).attr("fill", "none").attr("stroke", cssVar("--teacher")).attr("stroke-width", 2).attr("stroke-dasharray", "7 4");
    svg.append("path").attr("d", line<number>().x((_, i) => x(s + i)).y((d) => y(d))(forecast.q[L.indexOf(50)])).attr("fill", "none").attr("stroke", c).attr("stroke-width", 2.4);
    svg.append("line").attr("x1", x(forecast.origin + 0.5)).attr("x2", x(forecast.origin + 0.5)).attr("y1", 0).attr("y2", Hpx).attr("stroke", cssVar("--axis"));
  }
  if (tip < 0) svg.append("text").attr("x", W / 2).attr("y", Hpx / 2).attr("text-anchor", "middle").attr("class", "lbl").style("font-size", "16px").text("draw here, left to right");
  const items = [
    `<span><i style="border-color:var(--student)"></i>forecast (median)</span>`,
    `<span><i class="band" style="background:color-mix(in srgb, var(--student) 30%, transparent)"></i>40%, 80% and 98% intervals</span>`,
  ];
  if (truth || values.some((v, i) => i > origin && Number.isFinite(v))) items.push(`<span><i class="dash" style="border-color:var(--ink)"></i>what actually happened</span>`);
  if (showTeacher && teacher) items.push(`<span><i class="dash" style="border-color:var(--teacher)"></i>teacher (${mf.teacher.label}), median</span>`);
  if (origin - CONTEXT + 1 > 0) items.push(`<span class="muted">faded: older than the ${CONTEXT} points the model reads</span>`);
  $("legend").innerHTML = items.join("");
}

async function pump(): Promise<void> {
  if (busy || !dirty) return;
  dirty = false;
  const o = origin, ctx = values.slice(0, o + 1);
  if (o < 0 || ctx.filter(Number.isFinite).length < 16) { forecast = null; render(); return; }
  busy = true;
  try {
    const { q, ms } = await model.forecast(ctx);
    forecast = { origin: o, q };
    timed(ms);
  } finally { busy = false; }
  render();
  if (dirty) requestAnimationFrame(() => void pump());
}
const request = () => { dirty = true; requestAnimationFrame(() => void pump()); };

const idxAt = (cx: number) => { const r = pad.getBoundingClientRect(); return Math.round(x.invert(((cx - r.left) / r.width) * W)); };
const valAt = (cy: number) => { const r = pad.getBoundingClientRect(); return Math.max(0, Math.min(100, y.invert(((cy - r.top) / r.height) * Hpx))); };
let drawing = false, prev: [number, number] | null = null;
pad.addEventListener("pointerdown", (e) => {
  if (idxAt(e.clientX) >= DRAWABLE) return;
  drawing = true; prev = null; pad.setPointerCapture(e.pointerId);
  history.push([...values]); truth = null; setTeacher(null);
  stroke(e);
});
pad.addEventListener("pointermove", (e) => {
  if (drawing) return stroke(e);
  if (tip < 0) return;
  const i = Math.min(Math.max(idxAt(e.clientX), 0), tip);
  if (Number.isFinite(values[i]) && i !== origin) { origin = i; request(); }
});
pad.addEventListener("pointerup", () => { drawing = false; prev = null; });
pad.addEventListener("pointerleave", () => { if (!drawing && origin !== tip) { origin = tip; request(); } });
function stroke(e: PointerEvent): void {
  const i = Math.max(0, Math.min(DRAWABLE - 1, idxAt(e.clientX))), v = valAt(e.clientY);
  if (prev) { // fill every step between two pointer events by linear interpolation
    const [i0, v0] = prev;
    for (let k = Math.min(i0, i); k <= Math.max(i0, i); k++) values[k] = i === i0 ? v : v0 + ((v - v0) * (k - i0)) / (i - i0);
  } else values[i] = v;
  for (let k = i + 1; k < DRAWABLE; k++) values[k] = NaN; // the pen tip is "now": erase what was ahead of it
  prev = [i, v]; tip = origin = i; presetEl.value = "";
  render(); request();
}
function setTeacher(t: number[] | null): void { teacher = t; teacherBtn.disabled = !t; teacherBtn.title = t ? "" : "Available for the preset shapes"; }
function lastFinite(v: number[]): number { for (let k = v.length - 1; k >= 0; k--) if (Number.isFinite(v[k])) return k; return -1; }
function restore(v: number[]): void { values = v; tip = origin = lastFinite(v); truth = null; setTeacher(null); presetEl.value = ""; if (tip < 0) forecast = null; render(); request(); }
$("clear").onclick = () => { history.push([...values]); restore(new Array(DRAWABLE).fill(NaN)); };
$("undo").onclick = () => { if (history.length) restore(history.pop()!); };
teacherBtn.onclick = () => { showTeacher = !showTeacher; teacherBtn.setAttribute("aria-pressed", String(showTeacher)); render(); };

function scaled(s: Shape): (v: number) => number {
  const all = [...s.context, ...s.future].filter((v): v is number => v !== null);
  const lo = Math.min(...all), hi = Math.max(...all);
  return (v: number) => 10 + (80 * (v - lo)) / (hi - lo || 1);
}
function loadShape(name: string): void {
  const s = sh.shapes.find((z) => z.name === name);
  if (!s) return;
  const f = scaled(s);
  history.push([...values]);
  values = new Array(DRAWABLE).fill(NaN);
  s.context.forEach((c, i) => { values[i] = c === null ? NaN : f(c); });
  tip = origin = s.context.length - 1;
  truth = s.future.map(f);
  setTeacher(s.teacher_median.map(f));
  presetEl.value = name;
  render(); request();
}
presetEl.onchange = () => loadShape(presetEl.value);
for (const s of sh.shapes) presetEl.add(new Option(s.label, s.name));
loadShape(new URLSearchParams(location.search).get("shape") ?? "two_seasonalities");

// ---------------------------------------------------------------- accuracy vs size (fresh electricity)
{
  const naive = row("seasonal_naive").rel_wql;
  $("pareto-lede").textContent = `Error on ${fresh.n_windows} forecast windows from ${fresh.n_series} French electricity series, published after every model here was released. Each point is a model, with its 95% confidence interval. Lower is better; 1 is seasonal naive.`;
  const rows = fresh.rows.filter((r) => r.n_params > 0 && r.rel_wql < 1);
  const out = fresh.rows.filter((r) => r.n_params > 0 && r.rel_wql >= 1).map((r) => `${r.label} (${r.rel_wql.toFixed(2)})`);
  const Wp = 960, Hp = 420, m = { l: 52, r: 24, t: 16, b: 46 };
  const xs = scaleLog().domain([4e5, 6e8]).range([m.l, Wp - m.r]);
  const ys = scaleLinear().domain([Math.min(...rows.map((r) => r.ci[0])) - 0.02, Math.min(naive, Math.max(...rows.map((r) => r.ci[1])) + 0.03)]).nice().range([Hp - m.b, m.t]);
  const svg = select("#pareto").attr("viewBox", `0 0 ${Wp} ${Hp}`).attr("width", "100%");
  for (const t of ys.ticks(6)) {
    svg.append("line").attr("x1", m.l).attr("x2", Wp - m.r).attr("y1", ys(t)).attr("y2", ys(t)).attr("stroke", cssVar("--grid"));
    svg.append("text").attr("x", m.l - 8).attr("y", ys(t) + 4).attr("text-anchor", "end").text(t.toFixed(2));
  }
  for (const t of [1e6, 1e7, 1e8]) svg.append("text").attr("x", xs(t)).attr("y", Hp - m.b + 18).attr("text-anchor", "middle").text(fmtParams(t));
  svg.append("text").attr("x", (m.l + Wp - m.r) / 2).attr("y", Hp - 6).attr("text-anchor", "middle").attr("class", "lbl").text("parameters (log scale)");
  svg.append("text").attr("x", 14).attr("y", m.t + 4).attr("class", "lbl").attr("transform", `rotate(-90 14 ${m.t + 4})`).attr("text-anchor", "end").text("error vs seasonal naive");
  const LABEL = new Set(["student", "student-gt", "toto-2.0-313m", "toto-2.0-4m", "toto-2.0-22m", "timestral-28m", "chronos-bolt-tiny", "chronos-2", "t0-gamma"]);
  for (const r of rows) {
    const c = roleColor(r.role), cx = xs(r.n_params), cy = ys(r.rel_wql), ours = r.model === "student";
    const g = svg.append("g");
    g.append("line").attr("x1", cx).attr("x2", cx).attr("y1", ys(r.ci[0])).attr("y2", ys(r.ci[1])).attr("stroke", c).attr("stroke-width", 2).attr("opacity", 0.45);
    g.append("circle").attr("cx", cx).attr("cy", cy).attr("r", ours ? 7 : 5).attr("fill", c).attr("stroke", cssVar("--surface")).attr("stroke-width", 2);
    if (LABEL.has(r.model)) g.append("text").attr("x", cx + 10).attr("y", cy + 4).attr("class", ours ? "lbl-strong" : "lbl").style("font-size", ours ? "13.5px" : "12.5px").text(r.label);
    g.append("circle").attr("cx", cx).attr("cy", cy).attr("r", 14).attr("fill", "transparent")
      .on("mousemove", (e: MouseEvent) => showTip(e, `<b>${r.label}</b> · ${fmtParams(r.n_params)} params<br>${r.rel_wql.toFixed(3)} [${r.ci[0].toFixed(3)}, ${r.ci[1].toFixed(3)}]<br>80% interval covers ${Math.round(r.cov80 * 100)}% of outcomes`))
      .on("mouseleave", hideTip);
  }
  $("pareto-cap").innerHTML = `Blue: our student, and the same network trained on real futures instead of teacher forecasts (grey, 300k windows instead of 1M). Orange: the teacher. Green: other foundation models, used as released.
    ${out.length ? `Not shown, worse than seasonal naive: ${out.join(", ")}.` : ""} <a href="results.html">All numbers →</a>`;
}

// ---------------------------------------------------------------- where it works / fails (live forecasts)
{
  const sorted = [...sh.shapes].sort((a, b) => a.student_rel_wql - b.student_rel_wql);
  const pick = [...sorted.slice(0, 2), ...sorted.slice(-2)];
  const box = $("multiples");
  for (const [k, s] of pick.entries()) {
    const cell = document.createElement("div");
    cell.innerHTML = `<h4>${s.label} · ${s.student_rel_wql.toFixed(2)}</h4><p>${k < 2 ? "Best" : "Worst"}: ${s.description.toLowerCase()}.</p><svg></svg>`;
    box.append(cell);
    const ctx = s.context.map((v) => (v === null ? NaN : v));
    const { q, ms } = await model.forecast(ctx);
    timed(ms);
    const SHOW = 192, w = 300, h = 150, all = [...ctx.slice(-SHOW).filter(Number.isFinite), ...s.future, ...q[L.indexOf(1)], ...q[L.indexOf(99)]];
    const xs = scaleLinear().domain([0, SHOW + H]).range([0, w]), ys = scaleLinear().domain([Math.min(...all), Math.max(...all)]).range([h - 4, 4]);
    const svg = select(cell.querySelector("svg")!).attr("viewBox", `0 0 ${w} ${h}`).attr("width", "100%");
    const fx = (i: number) => xs(SHOW + i);
    for (const [a, b, op] of BANDS) svg.append("path").attr("d", area<number>().x((_, i) => fx(i)).y0((d) => ys(d)).y1((_, i) => ys(q[L.indexOf(b)][i]))(q[L.indexOf(a)])).attr("fill", cssVar("--student")).attr("opacity", op);
    svg.append("path").attr("d", line<number>().defined(Number.isFinite).x((_, i) => xs(i)).y((d) => ys(d))(ctx.slice(-SHOW))).attr("fill", "none").attr("stroke", cssVar("--ink-2")).attr("stroke-width", 1.3);
    svg.append("path").attr("d", line<number>().x((_, i) => fx(i)).y((d) => ys(d))(s.future)).attr("fill", "none").attr("stroke", cssVar("--ink")).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 2");
    svg.append("path").attr("d", line<number>().x((_, i) => fx(i)).y((d) => ys(d))(q[L.indexOf(50)])).attr("fill", "none").attr("stroke", cssVar("--student")).attr("stroke-width", 2);
  }
}
