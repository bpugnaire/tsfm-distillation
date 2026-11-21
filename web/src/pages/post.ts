// Shared script of every post: header, formulas, code cells from the repo, and the one widget (pinball minimiser).
import { scaleLinear } from "d3-scale";
import { select } from "d3-selection";
import { line } from "d3-shape";
import { cssVar, loadJSON, type Snippet } from "../lib/data";
import { header } from "../lib/nav";
import { highlightStatic, renderCode, renderMath } from "../lib/rich";

header();
renderMath();
highlightStatic();
const figs = document.querySelectorAll<HTMLElement>("figure.code[data-ref]");
if (figs.length) {
  const code = await loadJSON<Record<string, Snippet>>("code.json");
  for (const f of figs) {
    const c = code[f.dataset.ref!];
    if (c) renderCode(f, c);
    else f.innerHTML = `<figcaption>missing snippet: ${f.dataset.ref}</figcaption>`;
  }
}

// ---------------------------------------------------------------- widget: where does the pinball loss bottom out?
// The teacher's 13 quantiles of a log-normal, each weighted by the probability mass it stands for, used as observations.
// Move tau: the minimum of the expected pinball loss is the teacher's tau-quantile, i.e. the student copies the teacher.
const w = document.getElementById("w-kd");
if (w) {
  const LV = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99];
  // inverse normal CDF (Acklam's rational approximation)
  const ppf = (p: number): number => {
    const a = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924];
    const b = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857];
    const c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878];
    const d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742];
    const tail = (q: number) => { const r = Math.sqrt(-2 * Math.log(q)); return (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1); };
    if (p < 0.02425) return tail(p);
    if (p > 1 - 0.02425) return -tail(1 - p);
    const r = p - 0.5, s = r * r;
    return ((((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r) / (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1);
  };
  const vals = LV.map((p) => 11 * Math.exp(0.35 * ppf(p)));
  const edges = [0, ...LV.slice(1).map((v, i) => (v + LV[i]) / 2), 1];
  const mass = LV.map((_, i) => edges[i + 1] - edges[i]); // midpoint rule, sums to 1 (level_mass in our code)
  const tauEl = document.getElementById("w-kd-tau") as HTMLInputElement, out = document.getElementById("w-kd-out")!;
  const W = 680, H = 240, m = { l: 46, r: 12, t: 12, b: 36 };
  const svg = select("#w-kd-svg").attr("viewBox", `0 0 ${W} ${H}`).attr("width", "100%");
  const draw = () => {
    const tau = +tauEl.value;
    const grid = Array.from({ length: 301 }, (_, i) => 4 + (i * 20) / 300);
    const loss = grid.map((q) => vals.reduce((s, y, k) => s + mass[k] * (y >= q ? tau * (y - q) : (1 - tau) * (q - y)), 0));
    const iMin = loss.indexOf(Math.min(...loss));
    let acc = 0, qTau = vals[vals.length - 1];
    for (let k = 0; k < vals.length; k++) { acc += mass[k]; if (acc >= tau - 1e-12) { qTau = vals[k]; break; } }
    const x = scaleLinear().domain([4, 24]).range([m.l, W - m.r]);
    const y = scaleLinear().domain([0, Math.max(...loss)]).nice().range([H - m.b, m.t]);
    svg.selectAll("*").remove();
    for (const t of x.ticks(6)) svg.append("text").attr("x", x(t)).attr("y", H - m.b + 16).attr("text-anchor", "middle").text(t);
    svg.append("text").attr("x", W - m.r).attr("y", H - 4).attr("text-anchor", "end").attr("class", "lbl").text("student's forecast q for level τ →");
    svg.append("text").attr("x", m.l + 4).attr("y", m.t + 10).attr("class", "lbl").text("expected pinball loss under the teacher");
    vals.forEach((v, k) => svg.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", H - m.b).attr("y2", H - m.b - 6 - 110 * mass[k]).attr("stroke", cssVar("--teacher")).attr("stroke-width", 2.5));
    svg.append("path").attr("d", line<number>().x((_, i) => x(grid[i])).y((d) => y(d))(loss)).attr("fill", "none").attr("stroke", cssVar("--student")).attr("stroke-width", 2.2);
    svg.append("circle").attr("cx", x(grid[iMin])).attr("cy", y(loss[iMin])).attr("r", 5.5).attr("fill", cssVar("--student")).attr("stroke", cssVar("--surface")).attr("stroke-width", 2);
    out.innerHTML = `τ = <b>${tau.toFixed(2)}</b> · loss minimum at <b>${grid[iMin].toFixed(2)}</b> · teacher's τ-quantile <b>${qTau.toFixed(2)}</b>`;
  };
  tauEl.oninput = draw;
  draw();
}
