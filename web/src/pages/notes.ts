import { scaleLinear } from "d3-scale";
import { select } from "d3-selection";
import { line } from "d3-shape";
import { header } from "../lib/nav";
import { cssVar, loadJSON, type Gallery, type Results } from "../lib/data";

header();
const [res, gal] = await Promise.all([loadJSON<Results>("results.json"), loadJSON<Gallery>("gallery.json")]);
const $ = (id: string) => document.getElementById(id)!;

// deterministic RNG so the widget is the same for every reader
function mulberry32(a: number) { return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const rnd = mulberry32(7);
const gauss = () => Math.sqrt(-2 * Math.log(rnd() || 1e-12)) * Math.cos(2 * Math.PI * rnd());

// ---- W1: pinball loss minimiser ---------------------------------------------------------------
{
  const samples = Array.from({ length: 300 }, () => 10 * Math.exp(0.5 * gauss()));
  // inverse normal CDF (Acklam) for the teacher's lognormal quantiles
  const ppf = (p: number) => {
    const a = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924];
    const b = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857];
    const c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878];
    const d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742];
    const tail = (q: number) => { const r = Math.sqrt(-2 * Math.log(q)); return (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1); };
    if (p < 0.02425) return tail(p);
    if (p > 1 - 0.02425) return -tail(1 - p);
    const r = p - 0.5, s2 = r * r;
    return ((((((a[0] * s2 + a[1]) * s2 + a[2]) * s2 + a[3]) * s2 + a[4]) * s2 + a[5]) * r) / (((((b[0] * s2 + b[1]) * s2 + b[2]) * s2 + b[3]) * s2 + b[4]) * s2 + 1);
  };
  const tLevels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99];
  const tVals = tLevels.map((p) => 11 * Math.exp(0.35 * ppf(p)));
  const edges = [0, ...tLevels.slice(1).map((v, i) => (v + tLevels[i]) / 2), 1];
  const tW = tLevels.map((_, i) => edges[i + 1] - edges[i]);
  const W = 680, H = 240, m = { l: 44, r: 12, t: 12, b: 34 };
  const svg = select("#w1").attr("viewBox", `0 0 ${W} ${H}`).attr("width", "100%");
  const draw = () => {
    const tau = +($("w1-tau") as HTMLInputElement).value;
    $("w1-tauv").textContent = tau.toFixed(2);
    const teacher = ($("w1-src") as HTMLSelectElement).value === "teacher";
    const obs = teacher ? tVals : samples;
    const w = teacher ? tW : samples.map(() => 1 / samples.length);
    const grid = Array.from({ length: 301 }, (_, i) => 2 + (i * 28) / 300);
    const loss = grid.map((q) => obs.reduce((s, y, k) => s + w[k] * (y >= q ? tau * (y - q) : (1 - tau) * (q - y)), 0));
    const iMin = loss.indexOf(Math.min(...loss));
    // the tau-quantile of the (weighted) observations: smallest value whose cumulative weight reaches tau
    const ord = obs.map((y, k) => [y, w[k]] as const).sort((a, b) => a[0] - b[0]);
    let acc = 0, qTau = ord[ord.length - 1][0];
    for (const [y, wk] of ord) { acc += wk; if (acc >= tau - 1e-12) { qTau = y; break; } }
    const x = scaleLinear().domain([2, 30]).range([m.l, W - m.r]);
    const y = scaleLinear().domain([0, Math.max(...loss)]).nice().range([H - m.b, m.t]);
    svg.selectAll("*").remove();
    for (const t of y.ticks(4)) { svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(t)).attr("y2", y(t)).attr("stroke", cssVar("--grid")); svg.append("text").attr("x", m.l - 6).attr("y", y(t) + 4).attr("text-anchor", "end").text(t.toFixed(1)); }
    for (const t of x.ticks(7)) svg.append("text").attr("x", x(t)).attr("y", H - m.b + 14).attr("text-anchor", "middle").text(t);
    svg.append("text").attr("x", W - m.r).attr("y", H - 4).attr("text-anchor", "end").text("candidate forecast q →");
    svg.append("text").attr("x", m.l + 4).attr("y", m.t + 8).text("mean pinball loss");
    obs.forEach((v, k) => svg.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", H - m.b).attr("y2", H - m.b - (teacher ? 6 + 120 * w[k] : 6)).attr("stroke", cssVar("--teacher")).attr("stroke-width", teacher ? 2 : 1).attr("opacity", teacher ? 0.9 : 0.35));
    svg.append("path").attr("d", line<number>().x((_, i) => x(grid[i])).y((d) => y(d))(loss)).attr("fill", "none").attr("stroke", cssVar("--student")).attr("stroke-width", 2);
    svg.append("line").attr("x1", x(qTau)).attr("x2", x(qTau)).attr("y1", m.t).attr("y2", H - m.b).attr("stroke", cssVar("--ink-2")).attr("stroke-dasharray", "4 3");
    svg.append("circle").attr("cx", x(grid[iMin])).attr("cy", y(loss[iMin])).attr("r", 5).attr("fill", cssVar("--student")).attr("stroke", cssVar("--surface")).attr("stroke-width", 2);
    $("w1-cap").textContent = `Blue: mean pinball loss for each candidate q. Orange ticks: the observations${teacher ? " (height = probability mass of each teacher level)" : ""}. ` +
      `Minimum at q = ${grid[iMin].toFixed(2)}; the ${tau.toFixed(2)}-quantile of the observations (dashed) is ${qTau.toFixed(2)}.`;
  };
  $("w1-tau").addEventListener("input", draw);
  $("w1-src").addEventListener("change", draw);
  draw();
}

// ---- W2: CRPS bias table ---------------------------------------------------------------------------
{
  const b = res.crps_bias;
  const rows = Object.keys(b.equal_weights).map((k) => {
    const e = b.equal_weights[k], mw = b.mass_weights[k];
    const pct = (v: number) => `${v > b.exact ? "+" : ""}${(100 * (v / b.exact - 1)).toFixed(1)}%`;
    return `<tr><td>${k}</td><td class="num">${e.toFixed(3)}</td><td class="num">${pct(e)}</td><td class="num">${mw.toFixed(3)}</td><td class="num">${pct(mw)}</td></tr>`;
  }).join("");
  $("w2").innerHTML = `<table><thead><tr><th>levels</th><th class="num">equal weights</th><th class="num">bias</th><th class="num">mass weights</th><th class="num">bias</th></tr></thead><tbody>${rows}</tbody></table>
    <p class="small muted" style="margin:8px 0 0">Expected score of a perfectly calibrated N(0,1) forecast. Exact CRPS = 1/√π = ${b.exact.toFixed(3)}. Computed analytically in <code>scripts/build_web_data.py</code>.</p>`;
}

// ---- W3: splits + CI ---------------------------------------------------------------------------------
{
  const s = res.splits;
  const latest = Object.entries(s.releases).sort((a, b) => a[1].localeCompare(b[1])).pop()!;
  $("w3-splits").innerHTML = `train targets ≤ <b>${s.train_end}</b> · validation <b>${s.val_start} → ${s.val_end}</b> · fresh test ≥ <b>${s.fresh_start}</b>
    (latest release: ${latest[0]}, ${latest[1]}).`;
  const top = (suite: string) => res.suites[suite].rows[0];
  const f = top("fresh_fr_power"), sh = top("shapes");
  const wd = (r: typeof f) => r.rel_wql_ci[1] - r.rel_wql_ci[0];
  $("w3-ci").innerHTML = `Best model on each suite: fresh data (${f.n_windows} windows, but only ${f.n_clusters} series) has a CI ${wd(f).toFixed(3)} wide;
    the shapes suite (${sh.n_windows} fully independent series) has ${wd(sh).toFixed(3)}. More windows, no tighter interval: the effective sample size on fresh data is closer to the number of series.`;
  const half = res.suites.fresh_fr_power.rows.filter((r) => r.model !== "seasonal_naive").map(wd);
  $("ci-half").textContent = `±${(half.reduce((a, b) => a + b, 0) / half.length / 2).toFixed(3)}`;
}

// ---- W4: no model wins everywhere ------------------------------------------------------------
{
  const per = res.per_shape.shapes!;
  const fms = ["toto-2.0-4m", "toto-2.0-22m", "toto-2.0-313m", "t0-gamma", "chronos-2-small", "chronos-2"].filter((m) => per[m]);
  const shapes = Object.keys(per[fms[0]]);
  const losers = shapes.flatMap((s) => fms.filter((m) => per[m][s] > 1).map((m) => `${m} on <b>${s.replaceAll("_", " ")}</b>: ${per[m][s].toFixed(2)}× seasonal naive's error`));
  const wins: Record<string, number> = {};
  for (const s of shapes) { const best = fms.reduce((a, b) => (per[b][s] < per[a][s] ? b : a)); wins[best] = (wins[best] ?? 0) + 1; }
  $("w4").innerHTML = `<p style="margin-top:0">Foundation models <em>worse</em> than "repeat the last season":</p><ul>${losers.map((l) => `<li>${l}</li>`).join("")}</ul>
    <p>Shapes won (among foundation models): ${Object.entries(wins).sort((a, b) => b[1] - a[1]).map(([m, n]) => `${m} ${n}`).join(" · ")} (of ${shapes.length}).</p>
    <p class="muted" style="margin-bottom:0">Why it matters for distillation: a student trained to imitate one teacher inherits that teacher's blind spots. Choosing the teacher per window is a Phase 5 experiment.</p>`;
}

// ---- W5: robust vs naive scaling --------------------------------------------------------------------
{
  const sh = gal.shapes.find((s) => s.name === "spikes")!;
  const x = sh.context.filter((v): v is number => v !== null);
  const mean = x.reduce((a, b) => a + b, 0) / x.length;
  const sd = Math.sqrt(x.reduce((a, b) => a + (b - mean) ** 2, 0) / x.length);
  const sorted = [...x].sort((a, b) => a - b), med = sorted[Math.floor((x.length - 1) / 2)];
  const mad = x.reduce((a, b) => a + Math.abs(b - med), 0) / x.length;
  const zA = x.map((v) => (v - mean) / sd), zB = x.map((v) => Math.asinh((v - med) / mad));
  const W = 680, H = 200, m = { l: 34, r: 8, t: 18, b: 12 }, half = (W - 20) / 2;
  const svg = select("#w5").attr("viewBox", `0 0 ${W} ${H}`).attr("width", "100%");
  const panel = (z: number[], x0: number, title: string) => {
    const xs = scaleLinear().domain([0, z.length - 1]).range([x0 + m.l, x0 + half - m.r]);
    const ys = scaleLinear().domain([-6, 6]).range([H - m.b, m.t]);
    for (const t of [-4, 0, 4]) { svg.append("line").attr("x1", x0 + m.l).attr("x2", x0 + half - m.r).attr("y1", ys(t)).attr("y2", ys(t)).attr("stroke", cssVar("--grid")); svg.append("text").attr("x", x0 + m.l - 5).attr("y", ys(t) + 4).attr("text-anchor", "end").text(t); }
    svg.append("path").attr("d", line<number>().x((_, i) => xs(i)).y((d) => ys(Math.max(-6, Math.min(6, d))))(z)).attr("fill", "none").attr("stroke", cssVar("--student")).attr("stroke-width", 1);
    svg.append("text").attr("x", x0 + m.l).attr("y", 11).style("fill", cssVar("--ink-2")).text(title);
  };
  panel(zA, 0, "mean / std");
  panel(zB, half + 20, "asinh((x − median) / MAD)  ← ours");
  $("w5-cap").textContent = `The gallery's "spikes" context (${x.length} steps). mean/std: scale ${sd.toFixed(2)}, the daily wave spans about ±${(Math.max(...zA.map(Math.abs).filter((v) => v < 3))).toFixed(1)}. ` +
    `median/MAD: scale ${mad.toFixed(2)}, the wave fills the range and the spikes are squashed by asinh instead of dominating it.`;
}

// ---- W7: int8 ---------------------------------------------------------------------------------------
{
  const sens = (res.export as unknown as Record<string, { variants?: Record<string, { p99: number; max: number; size_mb: number }> }>)["int8_sensitivity_preview_gt_m"];
  const exp = res.export["preview_gt_m"];
  if (exp) { $("n-params").textContent = `${(exp.n_params / 1e6).toFixed(2)}M`; $("n-size").textContent = exp.fp32.size_mb.toFixed(1); }
  if (sens?.variants) {
    $("w7").innerHTML = `<table><thead><tr><th>what stays fp32</th><th class="num">p99 error</th><th class="num">max error</th><th class="num">file</th></tr></thead><tbody>` +
      Object.entries(sens.variants).map(([k, v]) => `<tr><td>${k}</td><td class="num">${(100 * v.p99).toFixed(1)}%</td><td class="num">${(100 * v.max).toFixed(1)}%</td><td class="num">${v.size_mb.toFixed(2)} MB</td></tr>`).join("") +
      `</tbody></table><p class="small muted" style="margin:8px 0 0">Preview student, 1000 stress-test windows (scales 1e-3 to 5e4, gaps, short series). Error = |int8 − fp32| / max(scale, |q − loc|). fp32 ONNX: ${exp ? exp.fp32.size_mb.toFixed(2) : "?"} MB.
      "Keep head fp32" changes nothing: the head is a <code>Gemm</code>, which the dynamic quantiser leaves in fp32 anyway. The patch embedding is what matters.</p>`;
  }
  const row = (suite: string, name: string) => res.suites[suite].rows.find((r) => r.model === name);
  const parts = ["fresh_fr_power", "shapes"].map((s) => {
    const a = row(s, "student-preview-gt-m"), b = row(s, "student-preview-gt-m-onnx-int8");
    return a && b ? `${s === "shapes" ? "shapes" : "fresh data"}: ${a.rel_wql.toFixed(4)} → ${b.rel_wql.toFixed(4)}` : "";
  }).filter(Boolean);
  $("w7-cap").innerHTML = `But a parity error of a few percent on stress-test windows is not an accuracy loss: relative WQL fp32 → int8, ${parts.join("; ")}.
    Measure the metric you care about, not only the parity.`;
}

// ---- cache estimate from measured throughput ------------------------------------------------------------
{
  const mps = Object.entries(res.throughput).filter(([k, v]) => v.device === "mps" && v.windows_per_s && !k.startsWith("toto-2.0-22m"));
  if (mps.length) $("cache-est").textContent = mps.map(([k, v]) => `${Math.round(300000 / v.windows_per_s! / 60)} min (${k.split("@")[0]})`).join(", ");
}
