import { scaleLinear } from "d3-scale";
import { select } from "d3-selection";
import { area, line } from "d3-shape";
import { header } from "../lib/nav";
import { cssVar, loadJSON, type Manifest, type Play, type PlayRound } from "../lib/data";
import { LEVELS, Student } from "../lib/ort";

header();
const ROUNDS = 8, SHOW = 192, H = 64, W = 960, Hpx = 340;
const svgEl = document.getElementById("pad") as unknown as SVGSVGElement;
const $ = (id: string) => document.getElementById(id)!;
const revealBtn = $("reveal") as HTMLButtonElement, nextBtn = $("next") as HTMLButtonElement;

const [play, mf] = await Promise.all([loadJSON<Play>("play.json"), loadJSON<Manifest>("models.json")]);
const entry = mf.models.find((m) => m.key === mf.default)!;
const student = await Student.load(`${entry.onnx}.fp32.onnx`);

// 4 electricity + 4 shapes, shuffled; a new mix every game
function pick(): PlayRound[] {
  const shuffle = <T,>(a: T[]) => { const b = [...a]; for (let i = b.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [b[i], b[j]] = [b[j], b[i]]; } return b; };
  const el = shuffle(play.rounds.filter((r) => r.kind === "electricity")).slice(0, ROUNDS / 2);
  const sh = shuffle(play.rounds.filter((r) => r.kind === "shape")).slice(0, ROUNDS / 2);
  return shuffle([...el, ...sh]);
}
let rounds = pick(), k = 0;
const tally = { you: 0, model: 0, tie: 0, yourErr: [] as number[], modelErr: [] as number[], teacherErr: [] as number[] };

// per-round state
let r: PlayRound, guess: number[], revealed = false, justRevealed = false, q: number[][] = [];
let x = scaleLinear(), y = scaleLinear(), lo = 0, hi = 1;

async function start(): Promise<void> {
  r = rounds[k];
  guess = new Array(H).fill(NaN);
  revealed = false;
  const ctx = r.context.map((v) => (v === null ? NaN : v));
  const shown = ctx.slice(-SHOW).filter(Number.isFinite);
  lo = Math.min(...shown); hi = Math.max(...shown);
  const pad = 0.35 * (hi - lo || 1);
  x = scaleLinear().domain([0, SHOW + H]).range([8, W - 8]);
  y = scaleLinear().domain([lo - pad, hi + pad]).range([Hpx - 6, 6]);
  $("title").textContent = r.title;
  $("meta").textContent = `round ${k + 1} of ${ROUNDS} · 1 step = ${r.step}${r.start ? ` · forecast starts ${r.start} UTC` : ""}`;
  $("result").innerHTML = "";
  revealBtn.style.display = ""; nextBtn.style.display = "none";
  q = (await student.forecast(ctx)).q; // computed now, shown at reveal
  board(); render(); coverage();
}

function board(): void {
  $("board").innerHTML = [
    ["Round", `${Math.min(k + 1, ROUNDS)} <small>/ ${ROUNDS}</small>`],
    ["You won", `${tally.you}`],
    ["Model won", `${tally.model}`],
  ].map(([l, v]) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
}

function render(): void {
  const svg = select(svgEl).attr("viewBox", `0 0 ${W} ${Hpx}`).attr("width", "100%");
  svg.selectAll("*").remove();
  const s0 = SHOW;
  svg.append("rect").attr("x", x(s0)).attr("width", x(SHOW + H) - x(s0)).attr("height", Hpx).attr("fill", cssVar("--accent-wash"));
  for (const t of y.ticks(4)) svg.append("line").attr("x1", 0).attr("x2", W).attr("y1", y(t)).attr("y2", y(t)).attr("stroke", cssVar("--grid"));
  const ctx = r.context.slice(-SHOW);
  svg.append("path").attr("d", line<number | null>().defined((d) => d !== null).x((_, i) => x(i)).y((d) => y(d as number))(ctx))
    .attr("fill", "none").attr("stroke", cssVar("--ink-2")).attr("stroke-width", 1.6);
  svg.append("line").attr("x1", x(s0 - 0.5)).attr("x2", x(s0 - 0.5)).attr("y1", 0).attr("y2", Hpx).attr("stroke", cssVar("--axis"));
  const fx = (i: number) => x(s0 + i);
  if (!revealed && guess.every((v) => !Number.isFinite(v)))
    svg.append("text").attr("x", (x(s0) + x(SHOW + H)) / 2).attr("y", Hpx / 2).attr("text-anchor", "middle").style("font-size", "15px").style("fill", cssVar("--ink-2")).text("draw the future here →");
  if (revealed) {
    const c = cssVar("--student"), lv = LEVELS.map((v) => Math.round(v * 100));
    svg.append("path").attr("d", area<number>().x((_, i) => fx(i)).y0((d) => y(d)).y1((_, i) => y(q[lv.indexOf(90)][i]))(q[lv.indexOf(10)])).attr("fill", c).attr("opacity", 0.2);
    svg.append("path").attr("d", line<number>().x((_, i) => fx(i)).y((d) => y(d))(r.teacher[1])).attr("fill", "none").attr("stroke", cssVar("--teacher")).attr("stroke-width", 1.8).attr("stroke-dasharray", "7 4");
    svg.append("path").attr("d", line<number>().x((_, i) => fx(i)).y((d) => y(d))(q[lv.indexOf(50)])).attr("fill", "none").attr("stroke", c).attr("stroke-width", 2.2);
    // the truth draws itself in
    const tp = svg.append("path").attr("d", line<number>().x((_, i) => fx(i)).y((d) => y(d))(r.future)).attr("fill", "none").attr("stroke", cssVar("--ink")).attr("stroke-width", 2);
    const len = (tp.node() as SVGPathElement).getTotalLength();
    const el = tp.node() as SVGPathElement;
    el.style.strokeDasharray = `${len} ${len}`; el.style.strokeDashoffset = `${len}`;
    if (!justRevealed) el.style.strokeDashoffset = "0";
    else requestAnimationFrame(() => { el.style.transition = "stroke-dashoffset .9s ease-out"; el.style.strokeDashoffset = "0"; });
  }
  svg.append("path").attr("d", line<number>().defined(Number.isFinite).x((_, i) => fx(i)).y((d) => y(d))(guess))
    .attr("fill", "none").attr("stroke", "#c026d3").attr("stroke-width", 2.6).attr("stroke-linecap", "round");
  const items = [`<span><i style="border-color:#c026d3"></i>your guess</span>`];
  if (revealed) items.push(`<span><i style="border-color:var(--ink)"></i>what happened</span>`, `<span><i style="border-color:var(--student)"></i>model</span>`, `<span><i class="dash" style="border-color:var(--teacher)"></i>teacher, ${play.teacher}</span>`);
  $("legend").innerHTML = items.join("");
}

// ---------------------------------------------------------------- drawing the guess
let drawing = false, prev: [number, number] | null = null;
const toIdx = (e: PointerEvent) => { const b = svgEl.getBoundingClientRect(); return Math.round(x.invert(((e.clientX - b.left) / b.width) * W)) - SHOW; };
const toVal = (e: PointerEvent) => { const b = svgEl.getBoundingClientRect(); return y.invert(((e.clientY - b.top) / b.height) * Hpx); };
svgEl.addEventListener("pointerdown", (e) => { if (revealed) return; drawing = true; prev = null; svgEl.setPointerCapture(e.pointerId); paint(e); });
svgEl.addEventListener("pointermove", (e) => { if (drawing) paint(e); });
svgEl.addEventListener("pointerup", () => { drawing = false; prev = null; });
function paint(e: PointerEvent): void {
  const i = Math.max(0, Math.min(H - 1, toIdx(e))), v = toVal(e);
  if (prev) { const [i0, v0] = prev; for (let j = Math.min(i0, i); j <= Math.max(i0, i); j++) guess[j] = i === i0 ? v : v0 + ((v - v0) * (j - i0)) / (i - i0); }
  else guess[i] = v;
  prev = [i, v];
  render(); coverage();
}
function coverage(): void {
  const c = guess.filter(Number.isFinite).length / H;
  $("coverage").textContent = revealed ? "" : c < 0.6 ? `draw across the shaded area (${Math.round(c * 100)}% so far)` : "ready";
  revealBtn.disabled = c < 0.6;
}
$("clear").onclick = () => { if (!revealed) { guess.fill(NaN); render(); coverage(); } };

// ---------------------------------------------------------------- scoring
function filled(g: number[]): number[] { // fill gaps in the guess: linear inside, flat at the ends
  const idx = g.map((v, i) => (Number.isFinite(v) ? i : -1)).filter((i) => i >= 0);
  return g.map((v, i) => {
    if (Number.isFinite(v)) return v;
    const a = [...idx].reverse().find((j) => j < i), b = idx.find((j) => j > i);
    if (a === undefined) return g[b!]; if (b === undefined) return g[a];
    return g[a] + ((g[b] - g[a]) * (i - a)) / (b - a);
  });
}
const err = (p: number[]) => { const span = (y.domain()[1] - y.domain()[0]) || 1; return (100 * r.future.reduce((s, t, i) => s + Math.abs(p[i] - t), 0)) / H / span; };

revealBtn.onclick = () => {
  guess = filled(guess); revealed = true;
  const lv = LEVELS.map((v) => Math.round(v * 100));
  const you = err(guess), mod = err(q[lv.indexOf(50)]), tch = err(r.teacher[1]);
  const inside = r.future.filter((t, i) => t >= q[lv.indexOf(10)][i] && t <= q[lv.indexOf(90)][i]).length / H;
  tally.yourErr.push(you); tally.modelErr.push(mod); tally.teacherErr.push(tch);
  const close = Math.abs(you - mod) < 0.1 * Math.max(you, mod);
  if (close) tally.tie++; else if (you < mod) tally.you++; else tally.model++;
  const verdict = close ? "Too close to call" : you < mod ? "You beat the model" : "The model wins this round";
  const why = r.kind === "electricity"
    ? "The models read three weeks of history; the chart shows eight days."
    : "";
  const max = Math.max(you, mod, tch) * 1.1;
  const bar = (label: string, v: number, c: string) => `<div class="scorebar"><span>${label}</span><div class="track"><div class="fill" style="background:${c}" data-w="${(100 * v) / max}"></div></div><span class="val">${v.toFixed(1)}%</span></div>`;
  $("result").innerHTML = `<div class="verdict pop">${verdict}</div>
    <div class="scores">${bar("You", you, "#c026d3")}${bar("Model (1M)", mod, cssVar("--student"))}${bar(`Teacher (${Math.round(play.teacher_params / 1e6)}M)`, tch, cssVar("--teacher"))}</div>
    <p class="why" style="margin-top:10px">The truth stayed inside the model's shaded band ${Math.round(inside * 100)}% of the time. ${why}</p>`;
  requestAnimationFrame(() => document.querySelectorAll<HTMLElement>(".fill").forEach((f) => { f.style.width = `${f.dataset.w}%`; }));
  revealBtn.style.display = "none"; nextBtn.style.display = ""; nextBtn.textContent = k + 1 < ROUNDS ? "Next round →" : "See final score →";
  justRevealed = true; render(); justRevealed = false; coverage(); board();
};
nextBtn.onclick = () => { k++; if (k < ROUNDS) void start(); else final(); };

function final(): void {
  const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
  const you = mean(tally.yourErr), mod = mean(tally.modelErr), tch = mean(tally.teacherErr);
  const msg = tally.you > tally.model ? "You out-forecast the model." : tally.you === tally.model ? "A draw." : "The model wins the match.";
  $("title").textContent = "Final score";
  $("meta").textContent = "";
  select(svgEl).selectAll("*").remove(); select(svgEl).attr("viewBox", "0 0 10 1");
  $("legend").innerHTML = "";
  $("result").innerHTML = `<div class="final pop"><div class="big">${tally.you} – ${tally.model}</div><p>you – model${tally.tie ? ` · ${tally.tie} too close to call` : ""}</p>
    <p class="verdict">${msg}</p><p class="why" style="margin:6px auto">Average error: you ${you.toFixed(1)}%, model ${mod.toFixed(1)}%, teacher ${tch.toFixed(1)}%.</p>
    <p class="row" style="justify-content:center;margin-top:16px"><button class="cta" id="again" style="border:0">Play again</button><a class="cta ghost" href="index.html">Draw your own →</a></p></div>`;
  nextBtn.style.display = "none"; revealBtn.style.display = "none";
  $("again").onclick = () => { rounds = pick(); k = 0; Object.assign(tally, { you: 0, model: 0, tie: 0, yourErr: [], modelErr: [], teacherErr: [] }); void start(); };
}

await start();
