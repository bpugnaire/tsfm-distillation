import { scaleLinear } from "d3-scale";
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
const OURS = "student-distilled";
const nice = (m: string) => (m === OURS ? "our student" : m);

const SUITES: [string, string, string][] = [
  ["shapes", "Synthetic shapes", "200 generated series: seasonality, trends, breaks, spikes."],
  ["fresh_fr_power", "French electricity", "13 series, published after every model was released."],
  ["multidomain_test", "Nine public datasets", "Weather, traffic, taxis, exchange rates. Some models were trained on these."],
];
const SMALL = 30e6; // "its size": up to ~30x the student
const POINT = new Set(["ttm-r2", "ttm-r2.1-l1"]); // point forecasts only: scored with all quantiles on the one line
const rowsOf = (s: string): SummaryRow[] => (res.suites[s]?.rows ?? []).filter((r) => !r.model.endsWith("-onnx-fp32") && !r.model.endsWith("-onnx-int8"));

// ---- small multiples: models up to 30M parameters, plus the best large model as a reference line ---------------
{
  const host = document.getElementById("small")!;
  const ours = rowsOf("shapes").find((r) => r.model === OURS);
  document.getElementById("small-lede")!.textContent = ours
    ? `Our student has ${fmtParams(ours.n_params)} parameters. Here it is against every forecasting model we tested with up to 30M, sorted by size. Bars show the 95% interval; the dashed line is the best large model. Models marked · give a single line with no uncertainty, which this score penalises; ▸ means off the scale.`
    : "";
  for (const [key, title, sub] of SUITES) {
    const all = rowsOf(key);
    if (!all.length) continue;
    const small = all.filter((r) => r.n_params && r.n_params <= SMALL && r.role !== "baseline").sort((a, b) => a.n_params - b.n_params);
    const big = all.filter((r) => r.n_params > SMALL).sort((a, b) => a.rel_wql - b.rel_wql)[0];
    const fig = document.createElement("div");
    fig.innerHTML = `<h3>${title}</h3><p class="sub">${sub}</p><svg></svg>`;
    host.append(fig);
    const W = 390, rowH = 26, m = { l: 168, r: 46, t: 6, b: 24 }, Hh = small.length * rowH + m.t + m.b;
    // point-only models (TTM) can sit far to the right: cap the axis just past the others and mark them off-scale
    const prob = small.filter((r) => !POINT.has(r.model));
    const vals = [...prob.flatMap((r) => r.rel_wql_ci), big?.rel_wql ?? 1];
    const x = scaleLinear().domain([Math.min(...vals) * 0.93, Math.max(...vals) * 1.03]).nice().range([m.l, W - m.r]).clamp(true);
    const off = (v: number) => v > x.domain()[1];
    const svg = select(fig.querySelector("svg") as SVGSVGElement).attr("viewBox", `0 0 ${W} ${Hh}`).attr("width", "100%");
    for (const t of x.ticks(4)) {
      svg.append("line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", m.t).attr("y2", Hh - m.b).attr("stroke", cssVar("--grid"));
      svg.append("text").attr("x", x(t)).attr("y", Hh - 6).attr("text-anchor", "middle").text(t.toFixed(2));
    }
    if (big) {
      svg.append("line").attr("x1", x(big.rel_wql)).attr("x2", x(big.rel_wql)).attr("y1", m.t).attr("y2", Hh - m.b).attr("stroke", cssVar("--ink-2")).attr("stroke-dasharray", "4 3");
      svg.append("rect").attr("x", x(big.rel_wql) - 6).attr("width", 12).attr("y", m.t).attr("height", Hh - m.t - m.b).attr("fill", "transparent")
        .on("mousemove", (e: MouseEvent) => showTip(e, `Best large model: <b>${big.model}</b> (${fmtParams(big.n_params)})<br>${big.rel_wql.toFixed(3)}`)).on("mouseleave", hideTip);
    }
    small.forEach((r, i) => {
      const yy = m.t + i * rowH + rowH / 2, isOurs = r.model === OURS, c = roleColor(r.role);
      if (isOurs) svg.append("rect").attr("x", 0).attr("y", yy - rowH / 2 + 1).attr("width", W).attr("height", rowH - 2).attr("rx", 5).attr("fill", cssVar("--accent-wash"));
      svg.append("text").attr("x", 6).attr("y", yy + 4).style("fill", isOurs ? cssVar("--ink") : cssVar("--ink-2")).style("font-size", "11.5px").style("font-weight", isOurs ? "650" : "400").text(nice(r.model) + (POINT.has(r.model) ? " ·" : ""));
      svg.append("text").attr("x", m.l - 8).attr("y", yy + 4).attr("text-anchor", "end").style("font-size", "11px").text(fmtParams(r.n_params));
      svg.append("line").attr("x1", x(r.rel_wql_ci[0])).attr("x2", x(r.rel_wql_ci[1])).attr("y1", yy).attr("y2", yy).attr("stroke", c).attr("stroke-width", 2.5).attr("opacity", 0.5).attr("stroke-linecap", "round");
      if (off(r.rel_wql)) svg.append("path").attr("d", `M${W - m.r - 2},${yy - 5} L${W - m.r + 6},${yy} L${W - m.r - 2},${yy + 5} Z`).attr("fill", c);
      else svg.append("circle").attr("cx", x(r.rel_wql)).attr("cy", yy).attr("r", isOurs ? 6 : 5).attr("fill", POINT.has(r.model) ? cssVar("--surface") : c).attr("stroke", POINT.has(r.model) ? c : cssVar("--surface")).attr("stroke-width", 2);
      svg.append("text").attr("x", W - 4).attr("y", yy + 4).attr("text-anchor", "end").style("fill", isOurs ? cssVar("--ink") : cssVar("--ink-2")).style("font-size", "12px").style("font-weight", isOurs ? "650" : "400").text(r.rel_wql.toFixed(2));
      svg.append("rect").attr("x", 0).attr("y", yy - rowH / 2).attr("width", W).attr("height", rowH).attr("fill", "transparent")
        .on("mousemove", (e: MouseEvent) => showTip(e, `<b>${nice(r.model)}</b> · ${fmtParams(r.n_params)} params<br>score ${r.rel_wql.toFixed(3)}, 95% interval ${r.rel_wql_ci[0].toFixed(3)} to ${r.rel_wql_ci[1].toFixed(3)}${r.latency_ms_t1_b1 ? `<br>${r.latency_ms_t1_b1.toFixed(1)} ms per forecast on CPU` : ""}`))
        .on("mouseleave", hideTip);
    });
  }
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
  const W = 920, H = 320, m = { l: 44, r: 170, t: 14, b: 32 };
  const svg = select("#ft").attr("viewBox", `0 0 ${W} ${H}`).attr("width", "100%");
  svg.selectAll("*").remove();
  const xs = ["none", "1w", "1m", "1y"], xl: Record<string, string> = { none: "no fine-tuning", "1w": "1 week", "1m": "1 month", "1y": "1 year" };
  const x = scaleLinear().domain([0, 3]).range([m.l + 30, W - m.r - 20]);
  const vals = [d.zero_shot_student ?? 1, ...Object.values(d.zoo), ...Object.values(d.curves).flatMap((c) => c.flatMap((p) => [p.min, p.max]))];
  const y = scaleLinear().domain([Math.min(...vals) * 0.92, Math.max(...vals) * 1.04]).nice().range([H - m.b, m.t]);
  for (const t of y.ticks(5)) {
    svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(t)).attr("y2", y(t)).attr("stroke", cssVar("--grid"));
    svg.append("text").attr("x", m.l - 8).attr("y", y(t) + 4).attr("text-anchor", "end").text(t.toFixed(1));
  }
  xs.forEach((k, i) => svg.append("text").attr("x", x(i)).attr("y", H - m.b + 18).attr("text-anchor", "middle").style("fill", cssVar("--ink-2")).text(xl[k]));
  const zl: SVGTextElement[] = [];
  for (const [name, v] of Object.entries(d.zoo)) {
    const c = cssVar("--comparator");
    svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(v)).attr("y2", y(v)).attr("stroke", c).attr("stroke-dasharray", "2 4").attr("stroke-width", 1.4);
    zl.push(svg.append("text").attr("x", W - m.r + 6).attr("y", y(v) + 4).style("fill", c).text(name).node()!);
  }
  zl.sort((a, b) => +a.getAttribute("y")! - +b.getAttribute("y")!);
  for (let i = 1; i < zl.length; i++) { const p = +zl[i - 1].getAttribute("y")!, c = +zl[i].getAttribute("y")!; if (c - p < 12) zl[i].setAttribute("y", String(p + 12)); }
  const series: [string, string, string][] = [["distilled", "from the distilled student", cssVar("--student")], ["scratch", "from random weights", cssVar("--baseline")]];
  for (const [key, label, c] of series) {
    const pts = d.curves[key] ?? [];
    const coords = [{ i: 0, mean: d.zero_shot_student, min: d.zero_shot_student, max: d.zero_shot_student }, ...pts.map((p) => ({ i: xs.indexOf(p.amount), mean: p.mean, min: p.min, max: p.max }))]
      .filter((p) => p.mean != null && (key === "distilled" || p.i > 0)) as { i: number; mean: number; min: number; max: number }[];
    const band = coords.map((p) => `${x(p.i)},${y(p.max)}`).join(" ") + " " + [...coords].reverse().map((p) => `${x(p.i)},${y(p.min)}`).join(" ");
    svg.append("polygon").attr("points", band).attr("fill", c).attr("opacity", 0.14);
    svg.append("polyline").attr("points", coords.map((p) => `${x(p.i)},${y(p.mean)}`).join(" ")).attr("fill", "none").attr("stroke", c).attr("stroke-width", 2.4);
    for (const p of coords) svg.append("circle").attr("cx", x(p.i)).attr("cy", y(p.mean)).attr("r", 5).attr("fill", c).attr("stroke", cssVar("--surface")).attr("stroke-width", 2)
      .on("mousemove", (e: MouseEvent) => showTip(e, `<b>${label}</b><br>${xl[xs[p.i]]}: ${p.mean.toFixed(3)}${p.min !== p.max ? ` (3 seeds: ${p.min.toFixed(3)} to ${p.max.toFixed(3)})` : ""}`)).on("mouseleave", hideTip);
    const at = coords.find((p) => p.i === (key === "scratch" ? 1 : 0));
    if (at) svg.append("text").attr("x", x(at.i) + 10).attr("y", y(at.mean) - 10).style("fill", c).style("font-weight", "600").style("font-size", "12px").text(label);
  }
  document.getElementById("ft-note")!.textContent = ftSuite === "fresh_fr_power"
    ? "The one-month point is worse than one week for both starts. We traced it to that month's unusual pattern plus overfitting."
    : "";
}
ft();

// ---- full leaderboard (folded) -----------------------------------------------------------------------------------
let suite = "fresh_fr_power";
function board(): void {
  document.getElementById("tabs")!.innerHTML = SUITES.filter(([k]) => res.suites[k])
    .map(([k, t]) => `<button class="chip" data-s="${k}" aria-pressed="${k === suite}">${t}</button>`).join("");
  document.querySelectorAll<HTMLButtonElement>("#tabs button").forEach((b) => { b.onclick = () => { suite = b.dataset.s!; board(); }; });
  const rs = rowsOf(suite);
  const body = rs.map((r, i) => `<tr${r.model === OURS ? ' style="background:var(--accent-wash)"' : ""}>
    <td class="num muted">${i + 1}</td>
    <td><span class="swatch" style="background:${roleColor(r.role)}"></span>${r.model === OURS ? "<b>our student</b>" : r.model}</td>
    <td class="num">${fmtParams(r.n_params)}</td>
    <td class="num"><b>${r.rel_wql.toFixed(3)}</b></td>
    <td class="num muted">${r.rel_wql_ci[0].toFixed(3)} to ${r.rel_wql_ci[1].toFixed(3)}</td>
    <td class="num">${r.latency_ms_t1_b1 != null ? r.latency_ms_t1_b1.toFixed(1) : "–"}</td></tr>`).join("");
  document.getElementById("board")!.innerHTML = `<table><thead><tr><th class="num">#</th><th>model</th><th class="num">params</th><th class="num">score</th><th class="num">95% interval</th><th class="num">ms on CPU</th></tr></thead><tbody>${body}</tbody></table>`;
}
board();
