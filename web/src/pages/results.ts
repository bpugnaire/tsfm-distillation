import { scaleLog } from "d3-scale";
import { select } from "d3-selection";
import { cssVar, fmtParams, hideTip, loadJSON, ratioWords, roleColor, showTip, type Finding, type Ratio, type Results } from "../lib/data";
import { header } from "../lib/nav";

header();
const $ = (id: string) => document.getElementById(id)!;
const res = await loadJSON<Results>("results.json");
const S = res.suites;
const row = (s: string, m: string) => S[s].rows.find((r) => r.model === m)!;
const g = res.gift;

// ---------------------------------------------------------------- intro
$("lede").innerHTML = `A 0.95M-parameter student distilled from Toto-2.0-313m scores ${row("fresh_fr_power", "student").rel_wql.toFixed(3)} on fresh
  French electricity (Toto-2.0-4m: ${row("fresh_fr_power", "toto-2.0-4m").rel_wql.toFixed(3)}, the teacher: ${row("fresh_fr_power", "toto-2.0-313m").rel_wql.toFixed(3)}).
  On GIFT-Eval it ranks ${g.rank} of ${g.n_models}. Below: the leaderboards, then every controlled comparison behind the result.`;
$("howto").innerHTML = `<b>Score</b>: weighted quantile loss (an approximation of CRPS, over the quantile levels 0.1 to 0.9) of each forecast window,
  divided by that of seasonal naive on the same window, then a geometric mean over windows. 0.70 means 30% less error than repeating the last season.
  <b>Comparisons</b> are paired: ratio of two models' scores on the same windows, with a 95% confidence interval from a bootstrap that resamples whole series.
  A comparison counts only if its interval excludes 1.`;

// ---------------------------------------------------------------- leaderboard
const TABS: [string, string, string][] = [
  ["fresh_fr_power", "Fresh electricity", `RTE éCO2mix: French national and regional demand, forecast windows whose target starts on or after ${res.splits.fresh_start}, after every model's release. Headline test set. ${S.fresh_fr_power.n_windows} windows, ${S.fresh_fr_power.n_series} series.`],
  ["shapes", "Synthetic shapes", `20 canonical shapes (seasonality, trends, level shifts, spikes, gaps…) × 10 random draws. ${S.shapes.n_windows} windows.`],
  ["multidomain_test", "Public data", `Held-out windows of public datasets (energy, traffic, weather, sales…). The teachers may have seen these series in pretraining: secondary. ${S.multidomain_test.n_windows} windows.`],
  ["gift", "GIFT-Eval", `The public GIFT-Eval benchmark, ${g.n_configs} short-term configurations, official harness and leaderboard rule (CRPS ratio to seasonal naive, geometric mean over configurations). Other models' scores are from their own leaderboard runs, with their own context length (Toto-2.0: 4096 steps); we re-ran Toto-2.0-4m at our 512.`],
];
let tab = "fresh_fr_power";
function tabs(): void {
  $("tabs").innerHTML = TABS.map(([k, l]) => `<button type="button" data-k="${k}" aria-pressed="${k === tab}">${l}</button>`).join("");
  $("tabs").querySelectorAll<HTMLButtonElement>("button").forEach((b) => (b.onclick = () => { tab = b.dataset.k!; tabs(); board(); }));
}
function board(): void {
  const t = TABS.find(([k]) => k === tab)!;
  $("suite-note").textContent = t[2];
  const rows = tab === "gift"
    ? g.rows.map((r) => ({ label: r.label, role: r.role, v: r.crps, ci: null as [number, number] | null, ours: r.role === "student", extra: r.mase != null ? r.mase.toFixed(3) : "–", n: "", ms: null as number | null }))
    : S[tab].rows.map((r) => ({ label: r.label, role: r.role, v: r.rel_wql, ci: r.ci as [number, number] | null, ours: r.model === "student", extra: `${Math.round(r.cov80 * 100)}%`, n: fmtParams(r.n_params), ms: r.ms }));
  const W = 960, rowH = 26, m = { l: 250, r: 70, t: 8, b: 30 }, H = m.t + m.b + rows.length * rowH;
  const lo = Math.min(...rows.map((r) => r.ci?.[0] ?? r.v)), hi = Math.max(...rows.map((r) => r.ci?.[1] ?? r.v));
  const x = scaleLog().domain([lo * 0.95, Math.max(hi, 1) * 1.05]).range([m.l, W - m.r]);
  const svg = select("#board").attr("viewBox", `0 0 ${W} ${H}`).attr("width", "100%");
  svg.selectAll("*").remove();
  for (const v of [0.4, 0.5, 0.6, 0.7, 0.8, 1, 1.25, 1.5, 2]) if (v >= x.domain()[0] && v <= x.domain()[1]) {
    svg.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", m.t).attr("y2", H - m.b).attr("stroke", cssVar(v === 1 ? "--axis" : "--grid")).attr("stroke-dasharray", v === 1 ? "4 3" : null);
    svg.append("text").attr("x", x(v)).attr("y", H - m.b + 16).attr("text-anchor", "middle").text(v === 1 ? "1 = seasonal naive" : v.toFixed(2));
  }
  rows.forEach((r, i) => {
    const yy = m.t + rowH * (i + 0.5), c = roleColor(r.role);
    if (r.ours) svg.append("rect").attr("x", 0).attr("y", yy - rowH / 2).attr("width", W).attr("height", rowH).attr("fill", cssVar("--wash"));
    svg.append("text").attr("x", m.l - 12).attr("y", yy + 4).attr("text-anchor", "end").attr("class", r.ours ? "lbl-strong" : "lbl").style("font-size", "13px").text(r.label);
    if (r.ci) svg.append("line").attr("x1", x(r.ci[0])).attr("x2", x(r.ci[1])).attr("y1", yy).attr("y2", yy).attr("stroke", c).attr("stroke-width", 3).attr("stroke-linecap", "round").attr("opacity", 0.4);
    svg.append("circle").attr("cx", x(r.v)).attr("cy", yy).attr("r", r.ours ? 6.5 : 5).attr("fill", c).attr("stroke", cssVar("--surface")).attr("stroke-width", 2);
    svg.append("text").attr("x", W - 6).attr("y", yy + 4).attr("text-anchor", "end").attr("class", r.ours ? "lbl-strong" : "lbl").style("font-size", "13px").text(r.v.toFixed(3));
  });
  const head = tab === "gift"
    ? `<th>model</th><th class="num">CRPS ratio</th><th class="num">MASE ratio</th>`
    : `<th>model</th><th class="num">params</th><th class="num">score</th><th class="num">95% interval</th><th class="num">80% interval coverage</th><th class="num">ms, CPU, 1 thread</th>`;
  const body = rows.map((r) => `<tr class="${r.ours ? "ours" : ""}"><td><span class="swatch" style="background:${roleColor(r.role)}"></span>${r.label}</td>${tab === "gift"
    ? `<td class="num">${r.v.toFixed(3)}</td><td class="num">${r.extra}</td>`
    : `<td class="num">${r.n}</td><td class="num">${r.v.toFixed(3)}</td><td class="num">${r.ci![0].toFixed(3)} – ${r.ci![1].toFixed(3)}</td><td class="num">${r.extra}</td><td class="num">${r.ms != null ? r.ms.toFixed(r.ms < 10 ? 2 : 0) : "–"}</td>`}</tr>`).join("");
  $("table").innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  const src = tab === "gift" ? g.source : S[tab].source;
  $("board-src").textContent = `Source: ${src.join(", ")}`
    + (tab === "gift" ? ` · our student ranks ${g.rank} of ${g.n_models} models on these configurations.` : " · latency: PyTorch for the foundation models, ONNX Runtime for our student, Apple M1 Pro.");
}
tabs();
board();

// ---------------------------------------------------------------- findings
function needle(r: Ratio): string {
  const W = 260, H = 30, x = scaleLog().domain([0.4, 2.5]).range([8, W - 8]).clamp(true);
  const w = ratioWords(r), c = w.cls === "good" ? cssVar("--good") : w.cls === "bad" ? cssVar("--bad") : cssVar("--baseline");
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px" aria-hidden="true">`;
  s += `<line x1="${x(1)}" x2="${x(1)}" y1="2" y2="${H - 11}" stroke="${cssVar("--axis")}" stroke-dasharray="3 3"/>`;
  for (const t of [0.5, 1, 2]) s += `<text x="${x(t)}" y="${H - 1}" text-anchor="middle" style="font-size:10px">${t === 1 ? "same" : `×${t}`}</text>`;
  s += `<line x1="${x(r[1])}" x2="${x(r[2])}" y1="10" y2="10" stroke="${c}" stroke-width="4" stroke-linecap="round" opacity=".5"/>`;
  s += `<circle cx="${x(r[0])}" cy="10" r="5.5" fill="${c}" stroke="${cssVar("--surface")}" stroke-width="2"/>`;
  if (r[0] > 2.5) s += `<text x="${W - 8}" y="14" text-anchor="end" style="font-size:10px">→ ×${r[0].toFixed(1)}</text>`;
  return s + "</svg>";
}
function card(f: Finding): string {
  const rows = f.contrasts.map((c) => {
    const w = ratioWords(c.ratio);
    return `<div class="contrast"><span>${c.name}</span>${needle(c.ratio)}<span class="w"><b class="${w.cls}">${w.text}</b>
      <span class="ci">ratio ${c.ratio[0].toFixed(3)} [${c.ratio[1].toFixed(3)}, ${c.ratio[2].toFixed(3)}]</span></span></div>`;
  }).join("");
  const src = [...new Set(f.contrasts.map((c) => c.source))].join(", ");
  return `<div class="finding"><h3>${f.title}</h3><p class="vs">${f.versus}</p>${rows}<p>${f.text}</p>
    <div class="do"><b>Do this:</b> ${f.takeaway}</div><div class="src">${src}</div></div>`;
}
for (const el of document.querySelectorAll<HTMLElement>(".findings[data-group]"))
  el.innerHTML = res.findings.filter((f) => f.group === el.dataset.group).map(card).join("");
document.querySelectorAll<SVGElement>(".contrast svg").forEach((s) => {
  s.addEventListener("mousemove", (e) => showTip(e as MouseEvent, "Dot: ratio of the two scores. Bar: 95% confidence interval. Left of the dashed line: the first model is better."));
  s.addEventListener("mouseleave", hideTip);
});

// ---------------------------------------------------------------- method
const rel = Object.entries(res.splits.releases).map(([k, v]) => `${k} ${v}`).join(", ");
$("method").innerHTML = [
  `<b>Student</b>: patch transformer, 4 layers, width 128, 4 heads, patches of 16 steps; reads 512 steps, forecasts 64, outputs 13 quantiles that cannot cross. 0.95M parameters.`,
  `<b>Training data</b> of the published student: 1M windows, half Gaussian-process synthetic series (KernelSynth), half from about 40 public datasets, filtered per series against GIFT-Eval. Labels: the teacher's quantiles, computed once and cached.`,
  `<b>No test leakage</b>: checkpoints are selected on a validation split, never on a test set. The fresh electricity test starts on ${res.splits.fresh_start}, after the model releases (${rel}); a unit test enforces the cut-off.`,
  `<b>One training seed</b> per student, except the fine-tuning runs (3 seeds). The intervals cover the choice of test windows, not training randomness: treat differences of 1–3% with care even when significant.`,
  `<b>Which student we publish</b>: the one taught by Toto-2.0-313m, because every model involved is Apache-2.0. We chose it after seeing the test scores, and it is the best of the four final students on fresh electricity, so its lead there over the other teachers' students is a selected number.`,
  `<b>Public data and GIFT-Eval</b> may overlap with what the teachers saw in pretraining. Our student's validation split contains earlier windows of 4 GIFT-Eval datasets (used for checkpoint selection only).`,
].map((s) => `<li>${s}</li>`).join("");
