import { header } from "../lib/nav";
import { cssVar, loadJSON, type Gallery } from "../lib/data";
import { drawFan } from "../lib/fan";
import { CONTEXT, LEVELS, Student } from "../lib/ort";

header();
const W = 960, Hpx = 340, LAYOUT_H = 64, M = 2;
const svg = document.getElementById("pad") as unknown as SVGSVGElement;
const hud = document.getElementById("hud")!;
const horizonEl = document.getElementById("horizon") as HTMLInputElement;
const compareEl = document.getElementById("compare") as HTMLInputElement;
const presetEl = document.getElementById("preset") as HTMLSelectElement;

let values: number[] = new Array(CONTEXT).fill(NaN);
const history: number[][] = [];
let truth: number[] | null = null;
let teacherMedian: number[] | null = null;
let forecastQ: number[][] | null = null;
const models: Record<string, Student> = {};
let current = "fp32";
const lat: number[] = [];
let busy = false, lastRun = 0, pending = false;

// pixel <-> (index, value). Values live in [0, 100]; the model does not care about units.
const xToIndex = (px: number) => ((px - M) / (W - 2 * M)) * (CONTEXT + LAYOUT_H);
const yToValue = (py: number) => 100 * (1 - (py - 4) / (Hpx - 8));

function render(): void {
  const h = +horizonEl.value;
  const q = forecastQ?.map((row) => row.slice(0, h));
  drawFan(svg, {
    context: values.map((v) => (Number.isFinite(v) ? v : null)),
    future: truth ?? undefined,
    q, levels: LEVELS, color: cssVar("--student"), width: W, height: Hpx, layoutH: LAYOUT_H,
    yDomain: [0, 100],
    extraMedian: compareEl.checked && teacherMedian ? { q: teacherMedian.slice(0, h), color: cssVar("--teacher") } : undefined,
  });
  // shaded drawing area
  const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  const xEnd = M + ((W - 2 * M) * CONTEXT) / (CONTEXT + LAYOUT_H);
  bg.setAttribute("x", "0"); bg.setAttribute("y", "0"); bg.setAttribute("width", String(xEnd)); bg.setAttribute("height", String(Hpx));
  bg.setAttribute("fill", cssVar("--accent-wash"));
  svg.insertBefore(bg, svg.firstChild);
}

async function runForecast(): Promise<void> {
  const m = models[current];
  if (!m || busy) { pending = true; return; }
  if (values.filter(Number.isFinite).length < 16) { forecastQ = null; render(); return; }
  busy = true; pending = false; lastRun = performance.now();
  try {
    const { q, ms } = await m.forecast(values);
    forecastQ = q;
    lat.push(ms); if (lat.length > 20) lat.shift();
    const med = [...lat].sort((a, b) => a - b)[Math.floor(lat.length / 2)];
    hud.textContent = `${current} · ${m.sizeMb.toFixed(2)} MB · ${m.threads} thread · median ${med.toFixed(1)} ms (last ${lat.length})`;
  } finally {
    busy = false;
  }
  render();
  if (pending) void runForecast();
}

function setAt(i: number, v: number): void { if (i >= 0 && i < CONTEXT) values[i] = Math.max(0, Math.min(100, v)); }

let drawing = false, prev: [number, number] | null = null;
svg.addEventListener("pointerdown", (e) => {
  drawing = true; prev = null; svg.setPointerCapture(e.pointerId);
  history.push([...values]); truth = null; teacherMedian = null; compareEl.disabled = true; compareEl.checked = false;
  onMove(e);
});
svg.addEventListener("pointermove", (e) => { if (drawing) onMove(e); });
svg.addEventListener("pointerup", () => { drawing = false; prev = null; void runForecast(); });

function onMove(e: PointerEvent): void {
  const r = svg.getBoundingClientRect();
  const px = ((e.clientX - r.left) / r.width) * W, py = ((e.clientY - r.top) / r.height) * Hpx;
  const i = xToIndex(px), v = yToValue(py);
  if (prev) { // fill every index between the previous and current pointer positions
    const [i0, v0] = prev;
    const a = Math.round(Math.min(i0, i)), b = Math.round(Math.max(i0, i));
    for (let k = a; k <= b; k++) setAt(k, b === a ? v : v0 + ((v - v0) * (k - i0)) / (i - i0 || 1));
  } else setAt(Math.round(i), v);
  prev = [i, v];
  render();
  if (performance.now() - lastRun > 100) void runForecast();
}

document.getElementById("clear")!.onclick = () => { history.push([...values]); values = new Array(CONTEXT).fill(NaN); truth = null; forecastQ = null; render(); };
document.getElementById("undo")!.onclick = () => { if (history.length) { values = history.pop()!; void runForecast(); render(); } };
document.getElementById("noise")!.onclick = () => {
  history.push([...values]);
  const obs = values.filter(Number.isFinite);
  const sd = 0.02 * (Math.max(...obs) - Math.min(...obs) || 10);
  values = values.map((v) => (Number.isFinite(v) ? Math.max(0, Math.min(100, v + sd * gauss())) : v));
  void runForecast();
};
document.getElementById("repeat")!.onclick = () => {
  const idx = values.map((v, i) => (Number.isFinite(v) ? i : -1)).filter((i) => i >= 0);
  if (idx.length < 4) return;
  history.push([...values]);
  const seg = values.slice(idx[0], idx[idx.length - 1] + 1);
  const out = new Array(CONTEXT).fill(NaN);
  for (let k = CONTEXT - 1, j = seg.length - 1; k >= 0; k--, j = (j - 1 + seg.length) % seg.length) out[k] = seg[j];
  values = out; truth = null; void runForecast();
};
function gauss(): number { return Math.sqrt(-2 * Math.log(Math.random() || 1e-9)) * Math.cos(2 * Math.PI * Math.random()); }

for (const id of ["fp32", "int8"]) {
  document.getElementById(`m-${id}`)!.onclick = async () => {
    current = id;
    for (const o of ["fp32", "int8"]) document.getElementById(`m-${o}`)!.setAttribute("aria-pressed", String(o === id));
    lat.length = 0;
    if (!models[id]) models[id] = await Student.load(`preview_gt_m.${id}.onnx`);
    void runForecast();
  };
}
horizonEl.oninput = () => { document.getElementById("hval")!.textContent = horizonEl.value; render(); };
compareEl.onchange = () => render();

// presets from the gallery (min-max mapped to 10..90 so they fit the pad)
let gallery: Gallery | null = null;
loadJSON<Gallery>("gallery.json").then((g) => {
  gallery = g;
  for (const s of g.shapes) presetEl.add(new Option(s.name.replaceAll("_", " "), s.name));
  const want = new URLSearchParams(location.search).get("shape");
  if (want) { presetEl.value = want; loadPreset(want); }
});
presetEl.onchange = () => loadPreset(presetEl.value);
function loadPreset(name: string): void {
  const s = gallery?.shapes.find((x) => x.name === name);
  if (!s || !gallery) return;
  const all = [...s.context, ...s.future].filter((v): v is number => v !== null);
  const lo = Math.min(...all), hi = Math.max(...all);
  const f = (v: number) => 10 + (80 * (v - lo)) / (hi - lo || 1);
  history.push([...values]);
  values = s.context.map((v) => (v === null ? NaN : f(v)));
  truth = s.future.map(f);
  const t = gallery.cells["toto-2.0-313m"]?.[name];
  teacherMedian = t ? t.q[gallery.levels.indexOf(0.5)].map(f) : null;
  compareEl.disabled = !teacherMedian;
  void runForecast();
}

render();
Student.load("preview_gt_m.fp32.onnx").then((m) => { models.fp32 = m; hud.textContent = "model ready · draw something"; void runForecast(); })
  .catch((e) => { hud.textContent = `model failed to load: ${e}`; });
