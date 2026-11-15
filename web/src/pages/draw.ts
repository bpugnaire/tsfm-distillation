import { scaleLinear } from "d3-scale";
import { select } from "d3-selection";
import { area, line } from "d3-shape";
import { header } from "../lib/nav";
import { cssVar, fmtParams, loadJSON, type Gallery, type Manifest } from "../lib/data";
import { CONTEXT, LEVELS, Student } from "../lib/ort";

header();
// The canvas spans STEPS time steps; you can draw on the first DRAWABLE of them, and the
// 64-step forecast always fits to the right of the pen tip.
const STEPS = 640, DRAWABLE = 576, H = 64, W = 960, Hpx = 360;
const svgEl = document.getElementById("pad") as unknown as SVGSVGElement;
const hint = document.getElementById("hint")!;
const presetEl = document.getElementById("preset") as HTMLSelectElement;
const cmpTeacher = document.getElementById("cmp-teacher") as HTMLButtonElement;
const x = scaleLinear().domain([0, STEPS]).range([0, W]);
const y = scaleLinear().domain([0, 100]).range([Hpx - 4, 4]);
const L = LEVELS.map((v) => Math.round(v * 100));

let values: number[] = new Array(DRAWABLE).fill(NaN);
const history: number[][] = [];
let lastDrawn = -1; // index of the pen tip / end of the drawing
let origin = -1; // forecast origin: pen tip while drawing, hovered point when backtesting
let truth: { start: number; v: number[] } | null = null; // a loaded shape's real future
let teacher: number[] | null = null; // precomputed teacher median for loaded shapes (origin = end only)
let forecast: { origin: number; q: number[][] } | null = null;
let model: Student | null = null;
let manifest: Manifest | null = null;
let showTeacher = false;
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
  const ln = line<number>().defined((d) => Number.isFinite(d)).x((_, i) => x(i)).y((d) => y(d));
  const o = origin;
  const past = values.map((v, i) => (i <= o ? v : NaN));
  const after = values.map((v, i) => (i >= o ? v : NaN));
  if (truth) svg.append("path").attr("d", line<number>().x((_, i) => x(truth!.start + i)).y((d) => y(d))(truth.v))
    .attr("fill", "none").attr("stroke", cssVar("--ink")).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 2");
  svg.append("path").attr("d", ln(after)).attr("fill", "none").attr("stroke", cssVar("--ink")).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 2");
  svg.append("path").attr("d", ln(past)).attr("fill", "none").attr("stroke", cssVar("--ink-2")).attr("stroke-width", 1.6);
  if (forecast && forecast.origin >= 0) {
    const s = forecast.origin + 1, c = cssVar("--student");
    for (const [a, b, op] of [[1, 99, 0.13], [10, 90, 0.25], [30, 70, 0.42]] as const) {
      const qa = forecast.q[L.indexOf(a)], qb = forecast.q[L.indexOf(b)];
      svg.append("path").attr("d", area<number>().x((_, i) => x(s + i)).y0((d) => y(d)).y1((_, i) => y(qb[i]))(qa)).attr("fill", c).attr("opacity", op);
    }
    if (showTeacher && teacher && forecast.origin === lastDrawn) {
      svg.append("path").attr("d", line<number>().x((_, i) => x(s + i)).y((d) => y(d))(teacher))
        .attr("fill", "none").attr("stroke", cssVar("--teacher")).attr("stroke-width", 2).attr("stroke-dasharray", "7 4");
    }
    svg.append("path").attr("d", line<number>().x((_, i) => x(s + i)).y((d) => y(d))(forecast.q[L.indexOf(50)]))
      .attr("fill", "none").attr("stroke", c).attr("stroke-width", 2.4);
    svg.append("line").attr("x1", x(forecast.origin + 0.5)).attr("x2", x(forecast.origin + 0.5)).attr("y1", 0).attr("y2", Hpx).attr("stroke", cssVar("--axis"));
    svg.append("circle").attr("cx", x(forecast.origin)).attr("cy", y(values[forecast.origin] ?? 50)).attr("r", 4.5).attr("fill", cssVar("--ink"))
      .attr("stroke", cssVar("--surface")).attr("stroke-width", 2);
  }
  if (lastDrawn < 0) svg.append("text").attr("x", W / 2).attr("y", Hpx / 2).attr("text-anchor", "middle").style("font-size", "15px").text("draw here, left to right");
  legend();
}

function legend(): void {
  const items = [`<span><i class="band" style="background:color-mix(in srgb, var(--student) 35%, transparent)"></i>forecast and its likely range</span>`];
  if (showTeacher && teacher) items.push(`<span><i class="dash" style="border-color:var(--teacher)"></i>teacher (${manifest?.teacher.label ?? "teacher"})</span>`);
  if (truth || values.some((v, i) => i > origin && Number.isFinite(v))) items.push(`<span><i class="dash" style="border-color:var(--ink)"></i>what actually happened</span>`);
  document.getElementById("legend")!.innerHTML = items.join("");
}

// ---------------------------------------------------------------- inference loop
function contextAt(o: number): number[] {
  return values.slice(Math.max(0, o - CONTEXT + 1), o + 1); // <= 512 past values, NaN = gap
}
async function pump(): Promise<void> {
  if (busy || !dirty || !model) return;
  dirty = false;
  const o = origin;
  const ctx = contextAt(o);
  if (o < 0 || ctx.filter(Number.isFinite).length < 16) { forecast = null; render(); return; }
  busy = true;
  try {
    const { q, ms } = await model.forecast(ctx);
    forecast = { origin: o, q: q.map((r) => r.slice(0, H)) };
    lat.push(ms); if (lat.length > 30) lat.shift();
    const med = [...lat].sort((a, b) => a - b)[Math.floor(lat.length / 2)];
    const el = document.getElementById("ms");
    if (el) el.textContent = `${med.toFixed(1)} ms`;
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
  history.push([...values]); truth = null; teacher = null; setTeacherAvailable(false);
  hint.style.opacity = "0";
  stroke(e);
});
svgEl.addEventListener("pointermove", (e) => {
  if (drawing) return stroke(e);
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
  presetEl.value = "";
  render();
  request();
}

// ---------------------------------------------------------------- buttons
function endOf(v: number[]): number { for (let k = v.length - 1; k >= 0; k--) if (Number.isFinite(v[k])) return k; return -1; }
function restore(v: number[]): void { values = v; lastDrawn = endOf(values); origin = lastDrawn; truth = null; teacher = null; setTeacherAvailable(false); presetEl.value = ""; request(); render(); }
document.getElementById("clear")!.onclick = () => { history.push([...values]); restore(new Array(DRAWABLE).fill(NaN)); forecast = null; hint.style.opacity = "1"; render(); };
document.getElementById("undo")!.onclick = () => { if (history.length) restore(history.pop()!); };
function setTeacherAvailable(ok: boolean): void {
  cmpTeacher.disabled = !ok;
  cmpTeacher.title = ok ? "The teacher's forecast from the end of the shape" : "Available on the preset shapes";
}
cmpTeacher.onclick = () => { showTeacher = !showTeacher; cmpTeacher.setAttribute("aria-pressed", String(showTeacher)); render(); };

// ---------------------------------------------------------------- preset shapes
let gallery: Gallery | null = null;
presetEl.onchange = () => loadPreset(presetEl.value);
document.getElementById("random")!.onclick = () => {
  if (!gallery) return;
  const others = gallery.shapes.filter((s) => s.name !== presetEl.value);
  const s = others[Math.floor(Math.random() * others.length)];
  presetEl.value = s.name; loadPreset(s.name);
};
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
  const t = manifest ? gallery.cells[manifest.teacher.name]?.[name] : undefined;
  teacher = t ? t.q[gallery.levels.indexOf(0.5)].map(f) : null;
  setTeacherAvailable(!!teacher);
  request(); render();
}

render();
const [mf, g] = await Promise.all([loadJSON<Manifest>("models.json"), loadJSON<Gallery>("gallery.json")]);
manifest = mf; gallery = g;
for (const s of g.shapes) presetEl.add(new Option(s.name.replaceAll("_", " "), s.name));
const entry = mf.models.find((z) => z.key === mf.default)!;
const ratio = entry.n_params ? Math.round(mf.teacher.n_params / entry.n_params) : null;
if (ratio) document.querySelector(".lede")!.textContent =
  `Draw a curve and it predicts what comes next, with its uncertainty. It learned by copying ${mf.teacher.label}, a forecasting model ${ratio} times its size.`;
document.getElementById("stats")!.innerHTML = [
  entry.n_params ? `<b>${fmtParams(entry.n_params)}</b> parameters` : "",
  entry.size_mb.fp32 ? `<b>${entry.size_mb.fp32.toFixed(1)} MB</b> download` : "",
  `<b id="ms">…</b> per forecast, here in your browser`,
].filter(Boolean).join('<span class="sep">·</span>');
model = await Student.load(`${entry.onnx}.fp32.onnx`);
const want = new URLSearchParams(location.search).get("shape") ?? "two_seasonalities";
presetEl.value = want; loadPreset(want);
