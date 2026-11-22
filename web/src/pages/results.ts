import { scaleLinear, scaleLog } from "d3-scale";
import { select } from "d3-selection";
import { cssVar, fmtParams, hideTip, loadJSON, ratioWords, roleColor, showTip, type Finding, type Ratio, type Results } from "../lib/data";
import { line, symbol, symbolCircle, symbolDiamond } from "d3-shape";
import { header } from "../lib/nav";

header();
const $ = (id: string) => document.getElementById(id)!;
const res = await loadJSON<Results>("results.json");
const S = res.suites;
const row = (s: string, m: string) => S[s].rows.find((r) => r.model === m)!;
const g = res.gift;

// ---------------------------------------------------------------- intro
$("lede").innerHTML = `A 0.87M-parameter student distilled from Timestral-5M scores ${row("fresh_fr_power", "student").rel_wql.toFixed(3)} on fresh
  French electricity (Toto-2.0-4m: ${row("fresh_fr_power", "toto-2.0-4m").rel_wql.toFixed(3)}, the teacher: ${row("fresh_fr_power", "timestral-5m").rel_wql.toFixed(3)}).
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

// ---------------------------------------------------------------- size against accuracy
const SZ = res.size_score;
const SZ_TABS: [string, string, string][] = [
  ["fresh_fr_power", "Fresh electricity", "score relative to seasonal naive"],
  ["shapes", "Synthetic shapes", "score relative to seasonal naive"],
  ["multidomain_test", "Public data", "score relative to seasonal naive"],
  ["gift", "GIFT-Eval", "CRPS relative to seasonal naive"],
];
let szTab = "fresh_fr_power";
interface Pt { label: string; n: number; v: number; kind: "student" | "student-mt" | "gt" | "teacher" | "zoo"; published?: boolean; hollow?: boolean; tip: string; link?: number }
function szPoints(k: string): Pt[] {
  const pts: Pt[] = [];
  for (const st of SZ.students) {
    const sc = st.scores[k];
    if (!sc) continue;
    const kind = !st.distilled ? "gt" : st.multi_token ? "student-mt" : "student";
    pts.push({ label: st.label, n: st.n_params, v: sc.score, kind, published: st.published,
      tip: `<b>${st.published ? "Published student: " : "Our student: "}${st.label}</b><br>${fmtParams(st.n_params)} parameters · score <b>${sc.score.toFixed(3)}</b>${sc.ci ? ` [${sc.ci[0].toFixed(3)}, ${sc.ci[1].toFixed(3)}]` : ""}<br><span class="src">${sc.source}</span>` });
  }
  for (const z of SZ.zoo) {
    const sc = z.scores[k];
    if (!sc) continue;
    const i = pts.length;
    pts.push({ label: z.label, n: z.n_params, v: sc.score, kind: z.teacher ? "teacher" : "zoo",
      tip: `<b>${z.label}</b>${z.teacher ? " (one of our teachers)" : ""}${z.gift_leakage && k === "gift" ? " · leaderboard flags test leakage" : ""}<br>${fmtParams(z.n_params)} parameters · score <b>${sc.score.toFixed(3)}</b><br><span class="src">${sc.source}</span>` });
    const s512 = z.scores[`${k}_512`];
    if (s512) pts.push({ label: `${z.label}, 512 steps`, n: z.n_params, v: s512.score, kind: z.teacher ? "teacher" : "zoo", hollow: true, link: i,
      tip: `<b>${z.label}</b> limited to 512 steps of history, like our students<br>score <b>${s512.score.toFixed(3)}</b><br><span class="src">${s512.source}</span>` });
  }
  return pts;
}
function szTabs(): void {
  $("sz-tabs").innerHTML = SZ_TABS.map(([k, l]) => `<button type="button" data-k="${k}" aria-pressed="${k === szTab}">${l}</button>`).join("");
  $("sz-tabs").querySelectorAll<HTMLButtonElement>("button").forEach((b) => (b.onclick = () => { szTab = b.dataset.k!; szTabs(); szChart(); }));
}
function szChart(): void {
  const pts = szPoints(szTab), W = 960, H = 440, m = { l: 56, r: 150, t: 16, b: 44 };
  const CAP = 1.0, onScale = pts.filter((p) => p.v <= CAP), off = pts.filter((p) => p.v > CAP);
  const x = scaleLog().domain([1e5, 5e8]).range([m.l, W - m.r]);
  const vs = onScale.map((p) => p.v), lo = Math.min(...vs), hi = Math.max(...vs);
  const pad = (hi - lo) * 0.08;
  const y = scaleLinear().domain([hi + pad, lo - pad]).range([H - m.b, m.t]); // lower error = higher on the chart
  const svg = select("#sz").attr("viewBox", `0 0 ${W} ${H}`).attr("width", "100%");
  svg.selectAll("*").remove();
  for (const t of y.ticks(6)) {
    svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(t)).attr("y2", y(t)).attr("stroke", cssVar("--grid"));
    svg.append("text").attr("x", m.l - 8).attr("y", y(t) + 4).attr("text-anchor", "end").text(t.toFixed(2));
  }
  for (const t of [1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8]) {
    svg.append("line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", m.t).attr("y2", H - m.b).attr("stroke", cssVar("--grid"));
    svg.append("text").attr("x", x(t)).attr("y", H - m.b + 16).attr("text-anchor", "middle").text(fmtParams(t));
  }
  svg.append("text").attr("x", (m.l + W - m.r) / 2).attr("y", H - 6).attr("text-anchor", "middle").attr("class", "lbl").text("parameters (log scale) →");
  svg.append("text").attr("x", m.l + 4).attr("y", m.t + 12).attr("class", "lbl").text(`↑ better · ${SZ_TABS.find(([k]) => k === szTab)![2]}`);
  // frontier: the best score at or below each size
  const sorted = onScale.filter((p) => !p.hollow).sort((a, b) => a.n - b.n);
  const front: Pt[] = [];
  for (const p of sorted) if (!front.length || p.v < front[front.length - 1].v) front.push(p);
  const stepPts: [number, number][] = [];
  front.forEach((p, i) => { stepPts.push([x(p.n), y(p.v)]); if (i < front.length - 1) stepPts.push([x(front[i + 1].n), y(p.v)]); });
  stepPts.push([W - m.r, y(front[front.length - 1].v)]);
  svg.append("path").attr("d", line()(stepPts) ?? "").attr("fill", "none").attr("stroke", cssVar("--ink-2")).attr("stroke-width", 1.2).attr("stroke-dasharray", "5 4").attr("opacity", 0.6);
  const COL: Record<Pt["kind"], string> = { student: cssVar("--student"), "student-mt": cssVar("--student"), gt: cssVar("--baseline"), teacher: cssVar("--teacher"), zoo: cssVar("--comparator") };
  for (const p of onScale) if (p.link != null) {
    const q = pts[p.link];
    if (q.v <= CAP) svg.append("line").attr("x1", x(q.n)).attr("x2", x(p.n)).attr("y1", y(q.v)).attr("y2", y(p.v)).attr("stroke", COL[p.kind]).attr("stroke-dasharray", "2 3");
  }
  const order = (p: Pt) => (p.kind === "zoo" || p.kind === "teacher" ? 0 : p.published ? 2 : 1);
  for (const p of [...onScale].sort((a, b) => order(a) - order(b))) {
    const c = COL[p.kind], g = svg.append("g").attr("transform", `translate(${x(p.n)},${y(p.v)})`);
    const sym = symbol().type(p.kind === "student-mt" ? symbolDiamond : symbolCircle).size(p.published ? 150 : p.kind.startsWith("student") || p.kind === "gt" ? 70 : 85);
    g.append("path").attr("d", sym() ?? "").attr("fill", p.hollow || p.kind === "gt" ? cssVar("--surface") : c).attr("stroke", p.hollow || p.kind === "gt" ? c : cssVar("--surface"))
      .attr("stroke-width", p.hollow || p.kind === "gt" ? 2 : 1.5).attr("opacity", p.kind.startsWith("student") && !p.published ? 0.8 : 1);
    if (p.published) g.append("circle").attr("r", 10).attr("fill", "none").attr("stroke", c).attr("stroke-width", 1.5);
    if (p.kind === "zoo" || p.kind === "teacher" || p.published) g.append("text").attr("x", 9).attr("y", 4).attr("class", p.published ? "lbl-strong" : "lbl").style("font-size", "12px").text(p.published ? "published student" : p.label);
    g.node()!.addEventListener("mousemove", (e) => showTip(e as MouseEvent, p.tip));
    g.node()!.addEventListener("mouseleave", hideTip);
  }
  const best = onScale.filter((p) => p.kind.startsWith("student")).sort((a, b) => a.v - b.v)[0];
  if (best && !best.published) svg.append("text").attr("x", x(best.n) + 10).attr("y", y(best.v) + 4).attr("class", "lbl-strong").style("font-size", "12px").text("best student");
  $("sz-legend").innerHTML = [
    [`<svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="${COL.student}"/></svg>`, "our distilled students"],
    [`<svg width="14" height="14"><path transform="translate(7,7)" d="${symbol().type(symbolDiamond).size(70)()}" fill="${COL.student}"/></svg>`, "…with the multi-token decoder"],
    [`<svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="none" stroke="${COL.gt}" stroke-width="2"/></svg>`, "same student trained on real futures only"],
    [`<svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="${COL.teacher}"/></svg>`, "foundation models we distilled from"],
    [`<svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="${COL.zoo}"/></svg>`, "other foundation models"],
    [`<i class="dash" style="border-color:${cssVar("--ink-2")}"></i>`, "best score at that size or smaller"],
  ].map(([i, l]) => `<span>${i} ${l}</span>`).join("");
  const offTxt = off.length ? ` Off the chart (worse than seasonal naive): ${off.map((p) => `${p.label} ${p.v.toFixed(2)}`).join(", ")}.` : "";
  $("sz-cap").innerHTML = (szTab === "gift"
    ? "GIFT-Eval, 55 short-term configurations, leaderboard rule. Foundation models use their own leaderboard context (up to thousands of steps); hollow points are re-runs with 512 steps of history, like our students. Students with a 2048-step context appear only here: our own suites at 2048 steps use other windows."
    : "Same windows for every model. Our students were scored by the study that trained them; foundation models by the benchmark.")
    + offTxt + " Hover a point for its numbers and source file.";
  const studs = SZ.students.filter((st) => st.scores[szTab]).sort((a, b) => a.scores[szTab].score - b.scores[szTab].score);
  $("sz-table").innerHTML = `<table><thead><tr><th>our student</th><th class="num">params</th><th class="num">score</th>${szTab === "gift" ? "" : `<th class="num">95% interval</th>`}<th>source</th></tr></thead><tbody>${studs.map((st) => {
    const sc = st.scores[szTab];
    return `<tr class="${st.published ? "ours" : ""}"><td>${st.label}${st.published ? " <b>(published)</b>" : ""}</td><td class="num">${fmtParams(st.n_params)}</td><td class="num">${sc.score.toFixed(3)}</td>${szTab === "gift" ? "" : `<td class="num">${sc.ci ? `${sc.ci[0].toFixed(3)} – ${sc.ci[1].toFixed(3)}` : "–"}</td>`}<td class="src">${sc.source}</td></tr>`;
  }).join("")}</tbody></table>`;
}
szTabs();
szChart();

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
  `<b>Student</b>: patch transformer, 4 layers, width 128, 4 heads, patches of 16 steps; reads 512 steps, forecasts 64 as 4 blocks of 16 (one query token each), outputs 13 quantiles that cannot cross. 0.87M parameters.`,
  `<b>Training data</b> of the published student: 3M windows, half Gaussian-process synthetic series (KernelSynth), half from about 40 public datasets, filtered per series against GIFT-Eval. Labels: Timestral-5M's quantiles, computed once and cached.`,
  `<b>No test leakage</b>: checkpoints are selected on a validation split, never on a test set. The fresh electricity test starts on ${res.splits.fresh_start}, after the model releases (${rel}); a unit test enforces the cut-off.`,
  `<b>One training seed</b> per student, except the fine-tuning runs (3 seeds). The intervals cover the choice of test windows, not training randomness: treat differences of 1–3% with care even when significant.`,
  `<b>Which student we publish</b>: the one with the best GIFT-Eval score among students under 1M parameters (3M training windows, Timestral-5M labels). GIFT-Eval is a test set, so its lead over the other students there is a selected number.`,
  `<b>Public data and GIFT-Eval</b> may overlap with what the teachers saw in pretraining. Our student's validation split contains earlier windows of 4 GIFT-Eval datasets (used for checkpoint selection only).`,
].map((s) => `<li>${s}</li>`).join("");
