import { scaleLinear } from "d3-scale";
import { select } from "d3-selection";
import { area, line } from "d3-shape";
import { header } from "../lib/nav";
import { cssVar, loadJSON, type Gallery } from "../lib/data";
import { CONTEXT, LEVELS, Student } from "../lib/ort";

header();
// The canvas spans STEPS time steps; you can draw on the first DRAWABLE of them, and the
// forecast (up to 64 steps) always fits to the right of the pen tip.
const STEPS = 640, DRAWABLE = 576, W = 960, Hpx = 360;
const svgEl = document.getElementById("pad") as unknown as SVGSVGElement;
const hud = document.getElementById("hud")!;
const horizonEl = document.getElementById("horizon") as HTMLInputElement;
const compareEl = document.getElementById("compare") as HTMLInputElement;
const presetEl = document.getElementById("preset") as HTMLSelectElement;
const x = scaleLinear().domain([0, STEPS]).range([0, W]);
const y = scaleLinear().domain([0, 100]).range([Hpx - 4, 4]);

let values: number[] = new Array(DRAWABLE).fill(NaN);
const history: number[][] = [];
let lastDrawn = -1; // index of the pen tip / end of the drawing
let origin = -1; // forecast origin: pen tip while drawing, hovered point when backtesting
let truth: { start: number; v: number[] } | null = null; // preset's real future
let teacher: number[] | null = null; // precomputed Toto 313m median for presets (origin = end only)
let forecast: { origin: number; q: number[][] } | null = null;
const models: Record<string, Student> = {};
let current = "fp32";
const lat: number[] = [];
let busy = false, dirty = false;

const idxAt = (clientX: number) => {
  const r = svgEl.getBoundingClientRect();
  return Math.round(x.invert(((clientX - r.left) / r.width) * W));
};
const valAt = (clientY: number) => {
  const r = svgEl.getBoundingClientRect();
  return Math.max(0, Math.min(100, y.invert(((clientY - r.top) / r.height) * Hpx)));
};

// ---------------------------------------------------------------- rendering
function render(): void {
  const svg = select(svgEl).attr("viewBox", `0 0 ${W} ${Hpx}`).attr("width", "100%");
  svg.selectAll("*").remove();
  svg.append("rect").attr("width", x(DRAWABLE)).attr("height", Hpx).attr("fill", cssVar("--accent-wash"));
  for (const t of [25, 50, 75]) svg.append("line").attr("x1", 0).attr("x2", W).attr("y1", y(t)).attr("y2", y(t)).attr("stroke", cssVar("--grid"));
  const ln = line<number>().defined((d) => Number.isFinite(d)).x((_, i) => x(i)).y((d) => y(d));
  const o = origin;
  // the drawn past (up to the origin) in ink; anything you drew after the origin is "the future" (dashed)
  const past = values.map((v, i) => (i <= o ? v : NaN));
  const after = values.map((v, i) => (i >= o ? v : NaN));
  if (truth) svg.append("path").attr("d", line<number>().x((_, i) => x(truth!.start + i)).y((d) => y(d))(truth.v))
    .attr("fill", "none").attr("stroke", cssVar("--ink")).attr("stroke-width", 1).attr("stroke-dasharray", "3 2");
  svg.append("path").attr("d", ln(after)).attr("fill", "none").attr("stroke", cssVar("--ink")).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 2");
  svg.append("path").attr("d", ln(past)).attr("fill", "none").attr("stroke", cssVar("--ink-2")).attr("stroke-width", 1.6);
  if (forecast && forecast.origin >= 0) {
    const h = +horizonEl.value, s = forecast.origin + 1, c = cssVar("--student");
    const L = LEVELS.map((v) => Math.round(v * 100));
    for (const [a, b, op] of [[1, 99, 0.15], [10, 90, 0.28], [30, 70, 0.45]] as const) {
      const qa = forecast.q[L.indexOf(a)].slice(0, h), qb = forecast.q[L.indexOf(b)].slice(0, h);
      svg.append("path").attr("d", area<number>().x((_, i) => x(s + i)).y0((d) => y(d)).y1((_, i) => y(qb[i]))(qa)).attr("fill", c).attr("opacity", op);
    }
    svg.append("path").attr("d", line<number>().x((_, i) => x(s + i)).y((d) => y(d))(forecast.q[L.indexOf(50)].slice(0, h)))
      .attr("fill", "none").attr("stroke", c).attr("stroke-width", 2.2);
    if (compareEl.checked && teacher && forecast.origin === lastDrawn) {
      svg.append("path").attr("d", line<number>().x((_, i) => x(s + i)).y((d) => y(d))(teacher.slice(0, h)))
        .attr("fill", "none").attr("stroke", cssVar("--teacher")).attr("stroke-width", 1.8).attr("stroke-dasharray", "6 3");
    }
    svg.append("line").attr("x1", x(forecast.origin + 0.5)).attr("x2", x(forecast.origin + 0.5)).attr("y1", 0).attr("y2", Hpx).attr("stroke", cssVar("--axis"));
    svg.append("circle").attr("cx", x(forecast.origin)).attr("cy", y(values[forecast.origin] ?? 50)).attr("r", 4).attr("fill", cssVar("--ink"));
  }
  if (lastDrawn < 0) svg.append("text").attr("x", W / 2).attr("y", Hpx / 2).attr("text-anchor", "middle").style("font-size", "15px").text("draw here, left to right");
}

// ---------------------------------------------------------------- inference loop
function contextAt(o: number): number[] {
  return values.slice(Math.max(0, o - CONTEXT + 1), o + 1); // <= 512 past values, NaN = gap
}
async function pump(): Promise<void> {
  if (busy || !dirty) return;
  const m = models[current];
  if (!m) return;
  dirty = false;
  const o = origin;
  const ctx = contextAt(o);
  if (o < 0 || ctx.filter(Number.isFinite).length < 16) { forecast = null; render(); return; }
  busy = true;
  try {
    const { q, ms } = await m.forecast(ctx);
    forecast = { origin: o, q };
    lat.push(ms); if (lat.length > 30) lat.shift();
    const med = [...lat].sort((a, b) => a - b)[Math.floor(lat.length / 2)];
    hud.textContent = `${current} · ${m.sizeMb.toFixed(2)} MB · ${m.threads} thread · inference median ${med.toFixed(1)} ms (last ${lat.length})`;
  } finally { busy = false; }
  render();
  if (dirty) requestAnimationFrame(() => void pump());
}
function request(): void { dirty = true; requestAnimationFrame(() => void pump()); }

// ---------------------------------------------------------------- drawing: the pen tip is "now"
let drawing = false, prev: [number, number] | null = null;
svgEl.addEventListener("pointerdown", (e) => {
  const i = idxAt(e.clientX);
  if (i >= DRAWABLE) return;
  drawing = true; prev = null; svgEl.setPointerCapture(e.pointerId);
  history.push([...values]); truth = null; teacher = null; compareEl.checked = false; compareEl.disabled = true;
  stroke(e);
});
svgEl.addEventListener("pointermove", (e) => {
  if (drawing) return stroke(e);
  // backtest by hover: forecast from any point of the drawing, using only what came before it
  if (lastDrawn < 0) return;
  const i = Math.min(Math.max(idxAt(e.clientX), 0), lastDrawn);
  if (Number.isFinite(values[i]) && i !== origin) { origin = i; request(); }
});
svgEl.addEventListener("pointerup", () => { drawing = false; prev = null; });
svgEl.addEventListener("pointerleave", () => { if (!drawing && origin !== lastDrawn) { origin = lastDrawn; request(); } });

function stroke(e: PointerEvent): void {
  const i = Math.max(0, Math.min(DRAWABLE - 1, idxAt(e.clientX))), v = valAt(e.clientY);
  if (prev) {
    const [i0, v0] = prev, a = Math.min(i0, i), b = Math.max(i0, i);
    for (let k = a; k <= b; k++) values[k] = b === a ? v : v0 + ((v - v0) * (k - i0)) / (i - i0);
  } else values[i] = v;
  for (let k = i + 1; k < DRAWABLE; k++) values[k] = NaN; // the pen tip is the present: erase what was ahead
  prev = [i, v];
  lastDrawn = i; origin = i;
  render();
  request();
}

// ---------------------------------------------------------------- buttons
function endOf(v: number[]): number { for (let k = v.length - 1; k >= 0; k--) if (Number.isFinite(v[k])) return k; return -1; }
function restore(v: number[]): void { values = v; lastDrawn = endOf(values); origin = lastDrawn; truth = null; teacher = null; request(); render(); }
document.getElementById("clear")!.onclick = () => { history.push([...values]); restore(new Array(DRAWABLE).fill(NaN)); forecast = null; render(); };
document.getElementById("undo")!.onclick = () => { if (history.length) restore(history.pop()!); };
document.getElementById("noise")!.onclick = () => {
  const obs = values.filter(Number.isFinite);
  if (!obs.length) return;
  history.push([...values]);
  const sd = 0.02 * (Math.max(...obs) - Math.min(...obs) || 10);
  restore(values.map((v) => (Number.isFinite(v) ? Math.max(0, Math.min(100, v + sd * gauss())) : v)));
};
document.getElementById("repeat")!.onclick = () => {
  const idx = values.map((v, i) => (Number.isFinite(v) ? i : -1)).filter((i) => i >= 0);
  if (idx.length < 4) return;
  history.push([...values]);
  const seg = values.slice(idx[0], idx[idx.length - 1] + 1);
  const out = new Array(DRAWABLE).fill(NaN);
  const end = Math.max(idx[idx.length - 1], 511);
  for (let k = end, j = seg.length - 1; k >= 0; k--, j = (j - 1 + seg.length) % seg.length) out[k] = seg[j];
  restore(out);
};
function gauss(): number { return Math.sqrt(-2 * Math.log(Math.random() || 1e-9)) * Math.cos(2 * Math.PI * Math.random()); }

for (const id of ["fp32", "int8"]) {
  document.getElementById(`m-${id}`)!.onclick = async () => {
    current = id;
    for (const o of ["fp32", "int8"]) document.getElementById(`m-${o}`)!.setAttribute("aria-pressed", String(o === id));
    lat.length = 0;
    if (!models[id]) models[id] = await Student.load(`preview_gt_m.${id}.onnx`);
    request();
  };
}
horizonEl.oninput = () => { document.getElementById("hval")!.textContent = horizonEl.value; render(); };
compareEl.onchange = () => render();

// ---------------------------------------------------------------- presets (gallery shapes)
let gallery: Gallery | null = null;
loadJSON<Gallery>("gallery.json").then((g) => {
  gallery = g;
  for (const s of g.shapes) presetEl.add(new Option(s.name.replaceAll("_", " "), s.name));
  const want = new URLSearchParams(location.search).get("shape");
  if (want) { presetEl.value = want; loadPreset(want); }
});
presetEl.onchange = () => loadPreset(presetEl.value);
function loadPreset(name: string): void {
  const s = gallery?.shapes.find((z) => z.name === name);
  if (!s || !gallery) return;
  const all = [...s.context, ...s.future].filter((v): v is number => v !== null);
  const lo = Math.min(...all), hi = Math.max(...all);
  const f = (v: number) => 10 + (80 * (v - lo)) / (hi - lo || 1);
  history.push([...values]);
  const v = new Array(DRAWABLE).fill(NaN);
  s.context.forEach((c, i) => { v[i] = c === null ? NaN : f(c); });
  values = v; lastDrawn = s.context.length - 1; origin = lastDrawn;
  truth = { start: lastDrawn + 1, v: s.future.map(f) };
  const t = gallery.cells["toto-2.0-313m"]?.[name];
  teacher = t ? t.q[gallery.levels.indexOf(0.5)].map(f) : null;
  compareEl.disabled = !teacher;
  request(); render();
}

render();
Student.load("preview_gt_m.fp32.onnx").then((m) => { models.fp32 = m; hud.textContent = "model ready · draw something"; request(); })
  .catch((e) => { hud.textContent = `model failed to load: ${e}`; });
