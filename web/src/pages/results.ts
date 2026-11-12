import { scaleLinear, scaleLog } from "d3-scale";
import { select } from "d3-selection";
import { header } from "../lib/nav";
import { cssVar, fmtParams, isPreview, loadJSON, roleColor, type Results, type SummaryRow } from "../lib/data";

header();
const res = await loadJSON<Results>("results.json");
const fresh = res.suites.fresh_fr_power.rows.filter((r) => !r.model.endsWith("-onnx-fp32"));
const shapes = res.suites.shapes.rows.filter((r) => !r.model.endsWith("-onnx-fp32"));
const tip = document.getElementById("tip")!;

function showTip(e: MouseEvent, html: string): void {
  tip.innerHTML = html; tip.style.display = "block";
  tip.style.left = `${Math.min(e.clientX + 12, innerWidth - 240)}px`; tip.style.top = `${e.clientY + 12}px`;
}
const hideTip = () => { tip.style.display = "none"; };

// ---- Pareto -------------------------------------------------------------
{
  const rows = fresh.filter((r) => r.latency_ms_t1_b1 && r.model !== "seasonal_naive");
  const W = 900, H = 380, m = { l: 52, r: 20, t: 14, b: 40 };
  const svg = select("#pareto").attr("viewBox", `0 0 ${W} ${H}`).attr("width", "100%");
  const x = scaleLog().domain([0.2, 2000]).range([m.l, W - m.r]);
  const ymax = Math.max(...rows.map((r) => r.rel_wql_ci[1])) * 1.05;
  const y = scaleLinear().domain([0.45, ymax]).nice().range([H - m.b, m.t]);
  for (const t of y.ticks(6)) {
    svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(t)).attr("y2", y(t)).attr("stroke", cssVar("--grid"));
    svg.append("text").attr("x", m.l - 8).attr("y", y(t) + 4).attr("text-anchor", "end").text(t.toFixed(2));
  }
  for (const t of [1, 10, 100, 1000]) {
    svg.append("line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", m.t).attr("y2", H - m.b).attr("stroke", cssVar("--grid"));
    svg.append("text").attr("x", x(t)).attr("y", H - m.b + 16).attr("text-anchor", "middle").text(`${t} ms`);
  }
  svg.append("text").attr("x", W - m.r).attr("y", H - 6).attr("text-anchor", "end").text("CPU latency, 1 thread, batch 1 (log scale) →");
  svg.append("text").attr("x", m.l).attr("y", m.t - 2).attr("dy", 10).attr("dx", 6).text("↓ relative WQL (lower is better)");
  for (const r of rows) {
    const cx = x(r.latency_ms_t1_b1!), c = roleColor(r.role);
    svg.append("line").attr("x1", cx).attr("x2", cx).attr("y1", y(r.rel_wql_ci[0])).attr("y2", y(r.rel_wql_ci[1])).attr("stroke", c).attr("stroke-width", 2).attr("opacity", 0.6);
    svg.append("circle").attr("cx", cx).attr("cy", y(r.rel_wql)).attr("r", 6).attr("fill", c).attr("stroke", cssVar("--surface")).attr("stroke-width", 2);
    svg.append("text").attr("x", cx + 9).attr("y", y(r.rel_wql) + 4).style("fill", cssVar("--ink-2")).text(r.model);
    svg.append("circle").attr("cx", cx).attr("cy", y(r.rel_wql)).attr("r", 14).attr("fill", "transparent")
      .on("mousemove", (e: MouseEvent) => showTip(e, `<b>${r.model}</b><br>rel WQL ${r.rel_wql.toFixed(3)} [${r.rel_wql_ci[0].toFixed(3)}, ${r.rel_wql_ci[1].toFixed(3)}]<br>${fmtParams(r.n_params)} params · ${r.latency_ms_t1_b1!.toFixed(1)} ms<br>80% coverage ${(100 * r.mean_cov80).toFixed(0)}%`))
      .on("mouseleave", hideTip);
  }
  document.getElementById("pareto-sub")!.textContent =
    `${fresh[0].n_windows} forecast windows from ${fresh[0].n_clusters} series. Colour = role (blue student, orange teacher, green comparator, gray baseline). ETS sits far right (statsforecast fits per series).`;
}

// ---- tables --------------------------------------------------------------
function table(el: string, rows: SummaryRow[]): void {
  const lo = 0.3, hi = 1.6;
  const sx = scaleLinear().domain([lo, hi]).range([0, 120]).clamp(true);
  const body = rows.map((r) => `<tr>
    <td><span class="swatch" style="background:${roleColor(r.role)}"></span>${r.model}${isPreview(r.model) ? ' <span class="tag">preview</span>' : ""}</td>
    <td class="num">${fmtParams(r.n_params)}</td>
    <td class="num">${r.rel_wql.toFixed(3)}</td>
    <td><svg width="124" height="14" style="vertical-align:middle"><line x1="${sx(1)}" x2="${sx(1)}" y1="0" y2="14" stroke="${cssVar("--axis")}"/><line x1="${sx(r.rel_wql_ci[0])}" x2="${sx(r.rel_wql_ci[1])}" y1="7" y2="7" stroke="${roleColor(r.role)}" stroke-width="2"/><circle cx="${sx(r.rel_wql)}" cy="7" r="3.5" fill="${roleColor(r.role)}"/></svg></td>
    <td class="num">${(100 * r.mean_cov80).toFixed(0)}%</td>
    <td class="num">${r.latency_ms_t1_b1 != null ? r.latency_ms_t1_b1.toFixed(1) : "–"}</td></tr>`).join("");
  document.getElementById(el)!.innerHTML = `<table><thead><tr><th>model</th><th class="num">params</th><th class="num">rel WQL</th><th>95% CI (line at 1 = naive)</th><th class="num">80% cov.</th><th class="num">ms</th></tr></thead><tbody>${body}</tbody></table>`;
}
table("t-fresh", fresh);
table("t-shapes", shapes);
document.getElementById("fresh-sub")!.textContent = `${fresh[0].n_windows} windows, ${fresh[0].n_clusters} series (national + 12 regions), targets from ${res.splits.fresh_start}. Nominal 80% coverage is 80%.`;

// ---- per-shape heat table (diverging around 1, log scale) -----------------
{
  const per = res.per_shape.shapes!;
  const models = Object.keys(per).filter((m) => !m.endsWith("-onnx-fp32") && m !== "seasonal_naive");
  const shapeNames = Object.keys(per[models[0]]);
  const dark = matchMedia("(prefers-color-scheme: dark)").matches && document.documentElement.dataset.theme !== "light";
  const blue = dark ? [57, 135, 229] : [42, 120, 214], red = dark ? [230, 103, 103] : [227, 73, 72], mid = dark ? [56, 56, 53] : [240, 239, 236];
  const col = (v: number) => {
    const t = Math.max(-1, Math.min(1, Math.log(v) / Math.log(4))); // +-1 at 4x better/worse
    const end = t < 0 ? blue : red, a = Math.abs(t);
    return `rgb(${mid.map((c, i) => Math.round(c + (end[i] - c) * a)).join(",")})`;
  };
  const txt = (v: number) => (Math.abs(Math.log(v) / Math.log(4)) > 0.55 ? "#ffffff" : cssVar("--ink"));
  const head = `<tr><th>shape</th>${models.map((m) => `<th class="num" style="font-size:11px">${m.replace("student-preview-gt-m", "student (preview)")}</th>`).join("")}</tr>`;
  const rows = shapeNames.map((s) => `<tr><td>${s.replaceAll("_", " ")}</td>${models.map((m) => {
    const v = per[m][s];
    return `<td class="num" style="background:${col(v)};color:${txt(v)};border-bottom:2px solid ${cssVar("--surface")}">${v.toFixed(2)}</td>`;
  }).join("")}</tr>`).join("");
  document.getElementById("heat")!.innerHTML = `<table>${head}${rows}</table>`;
}

// ---- splits --------------------------------------------------------------
{
  const s = res.splits;
  const rel = Object.entries(s.releases).map(([k, v]) => `<li>${k}: released ${v}</li>`).join("");
  document.getElementById("splits")!.innerHTML = `<p style="margin-top:0">The test data must be newer than every model, otherwise a model may have trained on it.</p>
  <ul><li><b>train</b> targets ≤ ${s.train_end}</li><li><b>validation</b> ${s.val_start} → ${s.val_end} (model selection only)</li>
  <li><b>fresh test</b> targets ≥ <b>${s.fresh_start}</b>, the day after the latest release below</li></ul><p class="muted">Model releases (Hugging Face):</p><ul>${rel}</ul>`;
}
