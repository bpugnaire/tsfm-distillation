import { scaleLinear, scaleLog } from "d3-scale";
import { select } from "d3-selection";
import { header } from "../lib/nav";
import { cssVar, fmtParams, loadJSON, roleColor, type FtSuite, type Results, type SummaryRow } from "../lib/data";

header();
const res = await loadJSON<Results & { finetune: Record<string, FtSuite> }>("results.json");
const tip = document.getElementById("tip")!;
const showTip = (e: MouseEvent, html: string) => {
  tip.innerHTML = html; tip.style.display = "block";
  tip.style.left = `${Math.min(e.clientX + 12, innerWidth - 250)}px`; tip.style.top = `${e.clientY + 12}px`;
};
const hideTip = () => { tip.style.display = "none"; };
const NICE: Record<string, string> = { "student-distilled": "our student · distilled", "student-gt-only": "our student · no teacher" };
const nice = (m: string) => NICE[m] ?? m;
const isOurs = (m: string) => m in NICE;

const SUITES: Record<string, { label: string; note: string }> = {
  fresh_fr_power: { label: "Fresh electricity (headline)", note: "French electricity demand (national + 12 regions), hourly, published after every model was released: nobody trained on it. The students never saw electricity at all: this is zero-shot for everyone." },
  multidomain_test: { label: "Multi-domain", note: "Held-out time windows of 9 public datasets (weather, traffic, taxis, exchange rates…). Caution: the foundation models may have seen these series in pretraining; our students trained on the earlier part of the same series." },
  shapes: { label: "Synthetic shapes", note: "200 generated series, 20 canonical shapes × 10 seeds (sine, trend breaks, spikes, chirps…). Seeds are disjoint from anything used in training." },
};
let suite = "fresh_fr_power";
const rows = () => res.suites[suite].rows.filter((r) => !r.model.endsWith("-onnx-fp32"));

function tabs(): void {
  document.getElementById("tabs")!.innerHTML = Object.entries(SUITES).filter(([k]) => res.suites[k])
    .map(([k, v]) => `<button class="chip" data-s="${k}" aria-pressed="${k === suite}">${v.label}</button>`).join("");
  document.querySelectorAll<HTMLButtonElement>("#tabs button").forEach((b) => { b.onclick = () => { suite = b.dataset.s!; draw(); }; });
  document.getElementById("suite-note")!.textContent = SUITES[suite].note;
}

// ---- Pareto: accuracy vs latency, labels decluttered ----------------------------------------------------------
function pareto(): void {
  const rs = rows().filter((r) => r.latency_ms_t1_b1 && r.model !== "seasonal_naive");
  const W = 920, H = 400, m = { l: 52, r: 24, t: 18, b: 40 };
  const svg = select("#pareto").attr("viewBox", `0 0 ${W} ${H}`).attr("width", "100%");
  svg.selectAll("*").remove();
  const x = scaleLog().domain([0.4, 1500]).range([m.l, W - m.r]);
  const ys = rs.map((r) => r.rel_wql_ci[1]).concat(rs.map((r) => r.rel_wql_ci[0]));
  const y = scaleLinear().domain([Math.min(...ys) * 0.95, Math.min(1.6, Math.max(...ys) * 1.03)]).nice().range([H - m.b, m.t]);
  for (const t of y.ticks(6)) {
    svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(t)).attr("y2", y(t)).attr("stroke", cssVar("--grid"));
    svg.append("text").attr("x", m.l - 8).attr("y", y(t) + 4).attr("text-anchor", "end").text(t.toFixed(2));
  }
  for (const t of [1, 10, 100, 1000]) {
    svg.append("line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", m.t).attr("y2", H - m.b).attr("stroke", cssVar("--grid"));
    svg.append("text").attr("x", x(t)).attr("y", H - m.b + 16).attr("text-anchor", "middle").text(`${t} ms`);
  }
  svg.append("text").attr("x", W - m.r).attr("y", H - 6).attr("text-anchor", "end").text("CPU time per forecast, 1 thread (log scale) →");
  svg.append("text").attr("x", m.l + 6).attr("y", m.t + 2).text("↓ relative WQL (lower is better)");
  // "better" corner hint
  svg.append("text").attr("x", m.l + 8).attr("y", H - m.b - 8).style("fill", cssVar("--comparator")).style("font-weight", "600").text("← faster and more accurate");
  const pts = rs.map((r) => ({ r, cx: x(r.latency_ms_t1_b1!), cy: y(r.rel_wql) }));
  for (const { r, cx } of pts) {
    const c = roleColor(r.role);
    svg.append("line").attr("x1", cx).attr("x2", cx).attr("y1", y(Math.min(r.rel_wql_ci[1], y.domain()[1]))).attr("y2", y(r.rel_wql_ci[0])).attr("stroke", c).attr("stroke-width", 2).attr("opacity", 0.55);
  }
  // selective direct labels: try right, left, above, below; skip (tooltip only) if every spot hits a dot or a label
  type Box = { x0: number; x1: number; y0: number; y1: number };
  const boxes: Box[] = pts.map((p) => ({ x0: p.cx - 7, x1: p.cx + 7, y0: p.cy - 7, y1: p.cy + 7 }));
  const hits = (b: Box) => boxes.some((o) => b.x0 < o.x1 && b.x1 > o.x0 && b.y0 < o.y1 && b.y1 > o.y0) || b.x1 > W - 4 || b.x0 < m.l;
  for (const { r, cx, cy } of [...pts].sort((a, b) => (isOurs(b.r.model) ? 1 : 0) - (isOurs(a.r.model) ? 1 : 0) || a.cy - b.cy)) {
    const ours = isOurs(r.model), label = nice(r.model), w = label.length * (ours ? 7 : 6.2), h = 13;
    const spots: [number, number, string][] = [[cx + 10, cy + 4, "start"], [cx - 10, cy + 4, "end"], [cx, cy - 11, "middle"], [cx, cy + 19, "middle"]];
    for (const [lx, ly, anchor] of spots) {
      const x0 = anchor === "start" ? lx : anchor === "end" ? lx - w : lx - w / 2;
      const b = { x0, x1: x0 + w, y0: ly - h + 3, y1: ly + 3 };
      if (hits(b)) continue;
      boxes.push(b);
      svg.append("text").attr("x", lx).attr("y", ly).attr("text-anchor", anchor).style("fill", ours ? cssVar("--ink") : cssVar("--ink-2"))
        .style("font-weight", ours ? "650" : "400").style("font-size", ours ? "12px" : "11px").text(label);
      break;
    }
  }
  for (const { r, cx, cy } of pts) {
    const c = roleColor(r.role), ours = isOurs(r.model);
    if (ours) svg.append("circle").attr("cx", cx).attr("cy", cy).attr("r", 11).attr("fill", c).attr("opacity", 0.15);
    svg.append("circle").attr("cx", cx).attr("cy", cy).attr("r", ours ? 7 : 5.5).attr("fill", r.model === "student-gt-only" ? cssVar("--surface") : c)
      .attr("stroke", r.model === "student-gt-only" ? c : cssVar("--surface")).attr("stroke-width", 2);
    svg.append("circle").attr("cx", cx).attr("cy", cy).attr("r", 15).attr("fill", "transparent")
      .on("mousemove", (e: MouseEvent) => showTip(e, `<b>${nice(r.model)}</b><br>rel WQL ${r.rel_wql.toFixed(3)} [${r.rel_wql_ci[0].toFixed(3)}, ${r.rel_wql_ci[1].toFixed(3)}]<br>${fmtParams(r.n_params)} params · ${r.latency_ms_t1_b1!.toFixed(1)} ms<br>80% interval coverage ${(100 * r.mean_cov80).toFixed(0)}%`))
      .on("mouseleave", hideTip);
  }
  const n = rows()[0];
  document.getElementById("pareto-sub")!.innerHTML =
    `${n.n_windows} forecast windows from ${n.n_clusters} series · colour = role: <span style="color:var(--student)">● our students</span> · <span style="color:var(--teacher)">● teachers</span> · <span style="color:var(--comparator)">● other foundation models</span> · <span style="color:var(--baseline)">● classical baselines</span>. ETS sits far right (fits per series). Students' latency is measured in Python on an Apple M1 Pro; the Draw page measures it live in your browser.`;
}

// ---- leaderboard ----------------------------------------------------------------------------------------------
function board(): void {
  const rs = rows();
  const lo = Math.min(0.3, ...rs.map((r) => r.rel_wql_ci[0])), hi = 1.6;
  const sx = scaleLog().domain([lo, hi]).range([4, 176]).clamp(true);
  const body = rs.map((r, i) => `<tr${isOurs(r.model) ? ' style="background:var(--accent-wash)"' : ""}>
    <td class="num muted">${i + 1}</td>
    <td><span class="swatch" style="background:${roleColor(r.role)}"></span>${isOurs(r.model) ? `<b>${nice(r.model)}</b>` : r.model}</td>
    <td class="num">${fmtParams(r.n_params)}</td>
    <td class="num"><b>${r.rel_wql.toFixed(3)}</b></td>
    <td><svg width="180" height="14" style="vertical-align:middle"><line x1="${sx(1)}" x2="${sx(1)}" y1="0" y2="14" stroke="${cssVar("--axis")}"/><line x1="${sx(r.rel_wql_ci[0])}" x2="${sx(r.rel_wql_ci[1])}" y1="7" y2="7" stroke="${roleColor(r.role)}" stroke-width="2"/><circle cx="${sx(r.rel_wql)}" cy="7" r="3.5" fill="${roleColor(r.role)}"/></svg></td>
    <td class="num">${(100 * r.mean_cov80).toFixed(0)}%</td>
    <td class="num">${r.latency_ms_t1_b1 != null ? r.latency_ms_t1_b1.toFixed(1) : "–"}</td></tr>`).join("");
  document.getElementById("board")!.innerHTML = `<table><thead><tr><th class="num">#</th><th>model</th><th class="num">params</th><th class="num">rel. WQL</th><th>95% interval (line = naive)</th><th class="num" title="Share of true values inside the 80% interval; 80% is ideal">80% cov.</th><th class="num">ms</th></tr></thead><tbody>${body}</tbody></table>`;
}

// ---- fine-tuning curves -----------------------------------------------------------------------------------------
let ftSuite = "fresh_fr_power";
function ft(): void {
  const all = res.finetune ?? {};
  const names: Record<string, string> = { fresh_fr_power: "Electricity", pedestrian_test: "Pedestrian counts" };
  document.getElementById("ft-tabs")!.innerHTML = Object.keys(all).map((k) => `<button class="chip" data-s="${k}" aria-pressed="${k === ftSuite}">${names[k] ?? k}</button>`).join("");
  document.querySelectorAll<HTMLButtonElement>("#ft-tabs button").forEach((b) => { b.onclick = () => { ftSuite = b.dataset.s!; ft(); }; });
  const d = all[ftSuite];
  if (!d) return;
  const W = 920, H = 340, m = { l: 52, r: 170, t: 14, b: 36 };
  const svg = select("#ft").attr("viewBox", `0 0 ${W} ${H}`).attr("width", "100%");
  svg.selectAll("*").remove();
  const xs = ["none", "1w", "1m", "1y"], xl: Record<string, string> = { none: "no fine-tuning", "1w": "1 week", "1m": "1 month", "1y": "1 year" };
  const x = scaleLinear().domain([0, 3]).range([m.l + 30, W - m.r - 20]);
  const vals = [d.zero_shot_student ?? 1, ...Object.values(d.zoo), ...Object.values(d.curves).flatMap((c) => c.flatMap((p) => [p.min, p.max]))];
  const y = scaleLinear().domain([Math.min(...vals) * 0.92, Math.max(...vals) * 1.04]).nice().range([H - m.b, m.t]);
  for (const t of y.ticks(5)) {
    svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(t)).attr("y2", y(t)).attr("stroke", cssVar("--grid"));
    svg.append("text").attr("x", m.l - 8).attr("y", y(t) + 4).attr("text-anchor", "end").text(t.toFixed(2));
  }
  xs.forEach((k, i) => svg.append("text").attr("x", x(i)).attr("y", H - m.b + 18).attr("text-anchor", "middle").style("fill", cssVar("--ink-2")).text(xl[k]));
  svg.append("text").attr("x", m.l + 6).attr("y", m.t + 4).text("↓ relative WQL");
  for (const [name, v] of Object.entries(d.zoo)) {
    const c = name.startsWith("timestral-209") ? cssVar("--teacher") : cssVar("--comparator");
    svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(v)).attr("y2", y(v)).attr("stroke", c).attr("stroke-dasharray", "2 4").attr("stroke-width", 1.5);
    svg.append("text").attr("x", W - m.r + 6).attr("y", y(v) + 4).style("fill", c).text(`${name} ${v.toFixed(2)}`);
  }
  // nudge zoo labels apart
  const labels = svg.selectAll<SVGTextElement, unknown>("text").nodes().filter((n) => +n.getAttribute("x")! === W - m.r + 6).sort((a, b) => +a.getAttribute("y")! - +b.getAttribute("y")!);
  for (let i = 1; i < labels.length; i++) { const p = +labels[i - 1].getAttribute("y")!, c = +labels[i].getAttribute("y")!; if (c - p < 12) labels[i].setAttribute("y", String(p + 12)); }
  const series: [string, string, string][] = [["distilled", "starting from the distilled student", cssVar("--student")], ["scratch", "starting from random weights", cssVar("--baseline")]];
  for (const [key, label, c] of series) {
    const pts = d.curves[key] ?? [];
    const coords = [{ i: 0, mean: d.zero_shot_student, min: d.zero_shot_student, max: d.zero_shot_student }, ...pts.map((p) => ({ i: xs.indexOf(p.amount), mean: p.mean, min: p.min, max: p.max }))]
      .filter((p) => p.mean != null && (key === "distilled" || p.i > 0)) as { i: number; mean: number; min: number; max: number }[];
    const band = coords.map((p) => `${x(p.i)},${y(p.max)}`).join(" ") + " " + [...coords].reverse().map((p) => `${x(p.i)},${y(p.min)}`).join(" ");
    svg.append("polygon").attr("points", band).attr("fill", c).attr("opacity", 0.14);
    svg.append("polyline").attr("points", coords.map((p) => `${x(p.i)},${y(p.mean)}`).join(" ")).attr("fill", "none").attr("stroke", c).attr("stroke-width", 2.4);
    for (const p of coords) svg.append("circle").attr("cx", x(p.i)).attr("cy", y(p.mean)).attr("r", 5).attr("fill", c).attr("stroke", cssVar("--surface")).attr("stroke-width", 2)
      .on("mousemove", (e: MouseEvent) => showTip(e, `<b>${label}</b><br>${xl[xs[p.i]]}: ${p.mean.toFixed(3)}${p.min !== p.max ? ` (seeds ${p.min.toFixed(3)}–${p.max.toFixed(3)})` : ""}`)).on("mouseleave", hideTip);
    const at = coords.find((p) => p.i === (key === "scratch" ? 1 : 0));
    if (at) svg.append("text").attr("x", x(at.i) + 10).attr("y", y(at.mean) - 10).style("fill", c).style("font-weight", "600").style("font-size", "12px").text(label);
  }
  document.getElementById("ft-note")!.textContent = ftSuite === "fresh_fr_power"
    ? "The 1-month point is worse than 1 week for both starts: we traced it to that particular month's pattern plus overfitting (Lab notes), not to a bug in the pipeline."
    : "Pedestrian counts: city sensor series held out from all distillation data.";
}

// ---- per-shape heat table (diverging around 1, log scale) -----------------------------------------------------
{
  const per = res.per_shape.shapes!;
  const models = Object.keys(per).filter((m) => !m.endsWith("-onnx-fp32") && m !== "seasonal_naive")
    .sort((a, b) => (isOurs(b) ? 1 : 0) - (isOurs(a) ? 1 : 0));
  const shapeNames = Object.keys(per[models[0]]);
  const dark = matchMedia("(prefers-color-scheme: dark)").matches && document.documentElement.dataset.theme !== "light";
  const blue = dark ? [57, 135, 229] : [42, 120, 214], red = dark ? [230, 103, 103] : [227, 73, 72], mid = dark ? [56, 56, 53] : [240, 239, 236];
  const col = (v: number) => {
    const t = Math.max(-1, Math.min(1, Math.log(v) / Math.log(4)));
    const end = t < 0 ? blue : red, a = Math.abs(t);
    return `rgb(${mid.map((c, i) => Math.round(c + (end[i] - c) * a)).join(",")})`;
  };
  const txt = (v: number) => (Math.abs(Math.log(v) / Math.log(4)) > 0.55 ? "#ffffff" : cssVar("--ink"));
  const head = `<tr><th>shape</th>${models.map((m) => `<th class="num" style="font-size:11px">${isOurs(m) ? `<b>${nice(m)}</b>` : m}</th>`).join("")}</tr>`;
  const body = shapeNames.map((s) => `<tr><td>${s.replaceAll("_", " ")}</td>${models.map((m) => {
    const v = per[m][s];
    return `<td class="num" style="background:${col(v)};color:${txt(v)};border-bottom:2px solid ${cssVar("--surface")}">${v.toFixed(2)}</td>`;
  }).join("")}</tr>`).join("");
  document.getElementById("heat")!.innerHTML = `<table>${head}${body}</table>`;
}

// ---- splits ----------------------------------------------------------------------------------------------------
{
  const s = res.splits;
  const rel = Object.entries(s.releases).map(([k, v]) => `<li>${k}: released ${v}</li>`).join("");
  document.getElementById("splits")!.innerHTML = `<p style="margin-top:0">The test data must be newer than every model, otherwise a model may have trained on it.</p>
  <ul><li><b>train</b> targets ≤ ${s.train_end}</li><li><b>validation</b> ${s.val_start} → ${s.val_end} (model selection only)</li>
  <li><b>fresh test</b> targets ≥ <b>${s.fresh_start}</b>, the day after the latest release below</li></ul><p class="muted">Model releases (Hugging Face):</p><ul>${rel}</ul>`;
}

function draw(): void { tabs(); pareto(); board(); }
draw();
ft();
export type { SummaryRow };
