import { scaleLinear } from "d3-scale";
import { select } from "d3-selection";
import { area, line } from "d3-shape";
import { header } from "../lib/nav";
import { cssVar, fmtParams, loadJSON, type Gallery, type Manifest, type Results } from "../lib/data";
import { CONTEXT, LEVELS, Student } from "../lib/ort";

header();
// The canvas spans STEPS time steps; you can draw on the first DRAWABLE of them, and the
// forecast (up to 64 steps) always fits to the right of the pen tip.
const STEPS = 640, DRAWABLE = 576, W = 960, Hpx = 360;
const svgEl = document.getElementById("pad") as unknown as SVGSVGElement;
const hud = document.getElementById("hud")!;
const hint = document.getElementById("hint")!;
const horizonEl = document.getElementById("horizon") as HTMLInputElement;
const presetEl = document.getElementById("preset") as HTMLSelectElement;
const cmpTeacher = document.getElementById("cmp-teacher") as HTMLButtonElement;
const cmpGt = document.getElementById("cmp-gt") as HTMLButtonElement;
const x = scaleLinear().domain([0, STEPS]).range([0, W]);
const y = scaleLinear().domain([0, 100]).range([Hpx - 4, 4]);
const L = LEVELS.map((v) => Math.round(v * 100));

let values: number[] = new Array(DRAWABLE).fill(NaN);
const history: number[][] = [];
let lastDrawn = -1; // index of the pen tip / end of the drawing
let origin = -1; // forecast origin: pen tip while drawing, hovered point when backtesting
let truth: { start: number; v: number[] } | null = null; // preset's real future
let teacher: number[][] | null = null; // precomputed teacher quantiles for presets (origin = end only)
let forecast: { origin: number; q: number[][] } | null = null;
let gtForecast: { origin: number; q: number[][] } | null = null;
const models: Record<string, Student> = {};
let precision: "fp32" | "int8" = "fp32";
const lat: number[] = [];
let busy = false, dirty = false;
let manifest: Manifest | null = null;
const show = { teacher: false, gt: false };

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
  const past = values.map((v, i) => (i <= o ? v : NaN));
  const after = values.map((v, i) => (i >= o ? v : NaN));
  if (truth) svg.append("path").attr("d", line<number>().x((_, i) => x(truth!.start + i)).y((d) => y(d))(truth.v))
    .attr("fill", "none").attr("stroke", cssVar("--ink")).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 2");
  svg.append("path").attr("d", ln(after)).attr("fill", "none").attr("stroke", cssVar("--ink")).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 2");
  svg.append("path").attr("d", ln(past)).attr("fill", "none").attr("stroke", cssVar("--ink-2")).attr("stroke-width", 1.6);
  if (forecast && forecast.origin >= 0) {
    const h = +horizonEl.value, s = forecast.origin + 1, c = cssVar("--student");
    for (const [a, b, op] of [[1, 99, 0.15], [10, 90, 0.28], [30, 70, 0.45]] as const) {
      const qa = forecast.q[L.indexOf(a)].slice(0, h), qb = forecast.q[L.indexOf(b)].slice(0, h);
      svg.append("path").attr("d", area<number>().x((_, i) => x(s + i)).y0((d) => y(d)).y1((_, i) => y(qb[i]))(qa)).attr("fill", c).attr("opacity", op);
    }
    const med = (q: number[][]) => q[L.indexOf(50)].slice(0, h);
    if (show.gt && gtForecast && gtForecast.origin === forecast.origin) {
      const g = gtForecast.q;
      for (const lv of [10, 90]) svg.append("path").attr("d", line<number>().x((_, i) => x(s + i)).y((d) => y(d))(g[L.indexOf(lv)].slice(0, h)))
        .attr("fill", "none").attr("stroke", cssVar("--baseline")).attr("stroke-width", 1).attr("stroke-dasharray", "2 3");
      svg.append("path").attr("d", line<number>().x((_, i) => x(s + i)).y((d) => y(d))(med(g)))
        .attr("fill", "none").attr("stroke", cssVar("--baseline")).attr("stroke-width", 2);
    }
    if (show.teacher && teacher && forecast.origin === lastDrawn) {
      svg.append("path").attr("d", line<number>().x((_, i) => x(s + i)).y((d) => y(d))(teacher[1].slice(0, h)))
        .attr("fill", "none").attr("stroke", cssVar("--teacher")).attr("stroke-width", 2).attr("stroke-dasharray", "7 4");
    }
    svg.append("path").attr("d", line<number>().x((_, i) => x(s + i)).y((d) => y(d))(med(forecast.q)))
      .attr("fill", "none").attr("stroke", c).attr("stroke-width", 2.4);
    svg.append("line").attr("x1", x(forecast.origin + 0.5)).attr("x2", x(forecast.origin + 0.5)).attr("y1", 0).attr("y2", Hpx).attr("stroke", cssVar("--axis"));
    svg.append("circle").attr("cx", x(forecast.origin)).attr("cy", y(values[forecast.origin] ?? 50)).attr("r", 4.5).attr("fill", cssVar("--ink"))
      .attr("stroke", cssVar("--surface")).attr("stroke-width", 2);
    if (forecast.origin < lastDrawn) svg.append("text").attr("x", x(forecast.origin) + 6).attr("y", 16).style("fill", cssVar("--ink-2")).text("backtest from here");
  }
  if (lastDrawn < 0) svg.append("text").attr("x", W / 2).attr("y", Hpx / 2).attr("text-anchor", "middle").style("font-size", "15px").text("draw here, left to right");
  legend();
}

function legend(): void {
  const items = [
    `<span><i class="band" style="background:linear-gradient(90deg, color-mix(in srgb, var(--student) 45%, transparent), color-mix(in srgb, var(--student) 15%, transparent))"></i>distilled student: median + 40 / 80 / 98% bands</span>`,
  ];
  if (show.gt) items.push(`<span><i style="border-color:var(--baseline)"></i>same student without the teacher (median, 80% dotted)</span>`);
  if (show.teacher && teacher) items.push(`<span><i class="dash" style="border-color:var(--teacher)"></i>teacher median (${manifest?.teacher.label ?? "teacher"})</span>`);
  if (truth || values.some((v, i) => i > origin && Number.isFinite(v))) items.push(`<span><i class="dash" style="border-color:var(--ink)"></i>what actually happened</span>`);
  document.getElementById("legend")!.innerHTML = items.join("");
}

// ---------------------------------------------------------------- inference loop
function contextAt(o: number): number[] {
  return values.slice(Math.max(0, o - CONTEXT + 1), o + 1); // <= 512 past values, NaN = gap
}
async function getModel(which: "main" | "gt"): Promise<Student | null> {
  if (!manifest) return null;
  const entry = which === "main" ? manifest.models.find((m) => m.key === manifest!.default) : manifest.models.find((m) => m.key === "student-gt-only");
  if (!entry) return null;
  const k = `${entry.onnx}.${precision}`;
  if (!models[k]) models[k] = await Student.load(`${k}.onnx`);
  return models[k];
}
async function pump(): Promise<void> {
  if (busy || !dirty) return;
  dirty = false;
  const o = origin;
  const ctx = contextAt(o);
  if (o < 0 || ctx.filter(Number.isFinite).length < 16) { forecast = null; render(); return; }
  busy = true;
  try {
    const m = await getModel("main");
    if (!m) return;
    const { q, ms } = await m.forecast(ctx);
    forecast = { origin: o, q };
    if (show.gt) { const g = await getModel("gt"); if (g) gtForecast = { origin: o, q: (await g.forecast(ctx)).q }; }
    lat.push(ms); if (lat.length > 30) lat.shift();
    const med = [...lat].sort((a, b) => a - b)[Math.floor(lat.length / 2)];
    hud.textContent = `${precision} · ${m.sizeMb.toFixed(1)} MB · ${m.threads} thread · ${med.toFixed(1)} ms per forecast`;
    const k = document.getElementById("kpi-ms");
    if (k) k.innerHTML = `${med.toFixed(1)} <small>ms</small>`;
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

for (const id of ["fp32", "int8"] as const) {
  document.getElementById(`m-${id}`)!.onclick = () => {
    precision = id;
    for (const o of ["fp32", "int8"]) document.getElementById(`m-${o}`)!.setAttribute("aria-pressed", String(o === id));
    lat.length = 0;
    request();
  };
}
function setTeacherAvailable(ok: boolean): void {
  cmpTeacher.disabled = !ok;
  cmpTeacher.title = ok ? "The teacher's forecast from the end of the shape (precomputed)" : "Load a shape: the teacher's forecast is precomputed for the shapes";
}
cmpTeacher.onclick = () => { show.teacher = !show.teacher; cmpTeacher.setAttribute("aria-pressed", String(show.teacher)); render(); };
cmpGt.onclick = () => { show.gt = !show.gt; cmpGt.setAttribute("aria-pressed", String(show.gt)); gtForecast = null; request(); };
horizonEl.oninput = () => { document.getElementById("hval")!.textContent = horizonEl.value; render(); };

// ---------------------------------------------------------------- presets (gallery shapes)
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
  const tname = manifest?.teacher.name ?? "timestral-209m";
  const t = gallery.cells[tname]?.[name];
  teacher = t ? [0.1, 0.5, 0.9].map((lv) => t.q[gallery!.levels.indexOf(lv)].map(f)) : null;
  setTeacherAvailable(!!teacher);
  request(); render();
}

// ---------------------------------------------------------------- headline numbers (all from data files)
function kpis(m: Manifest, res: Results | null): void {
  const s = m.models.find((z) => z.key === m.default)!;
  const ratio = s.n_params ? Math.round(m.teacher.n_params / s.n_params) : null;
  const sh = res?.suites.shapes?.rows;
  const stu = sh?.find((r) => r.model === s.key)?.rel_wql, tch = sh?.find((r) => r.model === m.teacher.name)?.rel_wql;
  const toto = sh?.find((r) => r.model === "toto-2.0-4m")?.rel_wql;
  const tiles = [
    [`${s.n_params ? fmtParams(s.n_params) : "–"} <small>params</small>`, ratio ? `${ratio}× smaller than its teacher, ${m.teacher.label}` : "parameters"],
    [`${s.size_mb.fp32?.toFixed(1) ?? "–"} <small>MB</small>`, `download (${s.size_mb.int8?.toFixed(1) ?? "–"} MB with int8 weights)`],
    [`<span id="kpi-ms">…</span>`, "per forecast, measured live in this tab (WASM, 1 thread)"],
    stu && tch ? [`${Math.round((stu / tch - 1) * 100)}% <small>behind</small>`, `its teacher on 200 synthetic shapes${toto ? `; ${Math.round((1 - stu / toto) * 100)}% ahead of Toto-2.0-4m (4M params)` : ""}`] : ["–", ""],
  ];
  document.getElementById("kpis")!.innerHTML = tiles.map(([v, l]) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
  if (s.teacher) document.getElementById("licence")!.textContent = `Teacher: ${m.teacher.label} (Mistral AI). ${m.teacher_licence}`;
  if (!m.models.some((z) => z.key === "student-gt-only")) cmpGt.disabled = true;
}

render();
const [mf, g, res] = await Promise.all([
  loadJSON<Manifest>("models.json"),
  loadJSON<Gallery>("gallery.json"),
  loadJSON<Results>("results.json").catch(() => null),
]);
manifest = mf; gallery = g;
for (const s of g.shapes) presetEl.add(new Option(s.name.replaceAll("_", " "), s.name));
kpis(mf, res);
getModel("main").then(() => {
  hud.textContent = "model ready";
  const want = new URLSearchParams(location.search).get("shape") ?? "two_seasonalities";
  presetEl.value = want; loadPreset(want);
}).catch((e) => { hud.textContent = `model failed to load: ${e}`; });
