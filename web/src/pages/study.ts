import { scaleLog } from "d3-scale";
import { header } from "../lib/nav";
import { cssVar, loadJSON } from "../lib/data";

interface Row { rel_wql: number; rel_wql_ci: [number, number]; cov80: number; vs_reference?: [number, number, number] }
interface Contrast { a: string; b: string; question: string; ratio: [number, number, number] }
interface SuiteTable { pairs?: Contrast[]; arms: Record<string, Row>; zero_shot_models: Record<string, { rel_wql: number; n_params: number }>; n_windows: number; flag: string | null }
interface Stage { study: string; reference_arm: string; suites: Record<string, SuiteTable> }
header();
const study = await loadJSON<Record<string, Stage>>("study.json");
const $ = (id: string) => document.getElementById(id)!;

const RQS: [string, string, string][] = [
  ["RQ1", "Which inputs should the teacher label? (real multi-domain, synthetic, teacher rollouts, noise)", "stage_a"],
  ["RQ2", "Teacher labels vs ground truth (α)?", "stage_a"],
  ["RQ3", "Which teacher(s)? Single, ensemble, per-window weighting", "stage_b"],
  ["RQ4", "Which distillation loss and weighting?", "stage_b"],
  ["RQ5", "Scaling: data size, student size, rollout length", "stage_c"],
  ["RQ6", "How much fine-tuning on your data closes the gap to the big models?", "stage_d"],
];
$("rq").innerHTML = RQS.map(([id, q, st]) => `<tr><td><b>${id}</b></td><td>${q}</td><td>${st.replace("_", " ")}</td><td>${study[st] ? "results below" : '<span class="muted">not run</span>'}</td></tr>`).join("");

const SUITE_NAMES: Record<string, string> = { shapes: "Shapes (synthetic, 200 series)", multidomain_test: "Held-out multi-domain test", fresh_fr_power: "Fresh French electricity (zero-shot for everyone)" };
let html = "";
for (const [name, st] of Object.entries(study)) {
  html += `<h2 id="${name}">${name.replace("_", " ").toUpperCase()}</h2><p class="small muted">Reference arm: <code>${st.reference_arm}</code>. One seed (exploratory): differences smaller than seed-to-seed noise need confirmation.</p>`;
  for (const [suite, t] of Object.entries(st.suites)) {
    const arms = Object.entries(t.arms).filter(([m]) => m !== "seasonal_naive").sort((a, b) => a[1].rel_wql - b[1].rel_wql);
    const teach = Object.entries(t.zero_shot_models).filter(([m]) => m !== "ets").sort((a, b) => a[1].rel_wql - b[1].rel_wql);
    // needle chart: paired ratio vs reference on a log scale
    const W = 640, rowH = 22, H = arms.length * rowH + 30, ml = 150;
    const x = scaleLog().domain([0.5, 2]).range([ml, W - 20]);
    let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%"><line x1="${x(1)}" x2="${x(1)}" y1="4" y2="${H - 22}" stroke="${cssVar("--ink")}" stroke-dasharray="3 3"/>`;
    for (const t of [0.5, 0.75, 1, 1.5, 2]) svg += `<text x="${x(t)}" y="${H - 6}" text-anchor="middle">${t}</text>`;
    arms.forEach(([m, r], i) => {
      const y = 14 + i * rowH;
      svg += `<text x="${ml - 8}" y="${y + 4}" text-anchor="end" style="fill:${cssVar("--ink-2")}">${m}</text>`;
      if (r.vs_reference) {
        const [p, lo, hi] = r.vs_reference.map((v) => Math.min(2, Math.max(0.5, v)));
        const sig = r.vs_reference[2] < 1 || r.vs_reference[1] > 1;
        const c = r.vs_reference[2] < 1 ? cssVar("--student") : r.vs_reference[1] > 1 ? cssVar("--teacher") : cssVar("--baseline");
        svg += `<line x1="${x(lo)}" x2="${x(hi)}" y1="${y}" y2="${y}" stroke="${c}" stroke-width="3"/><circle cx="${x(p)}" cy="${y}" r="${sig ? 5 : 4}" fill="${c}"/>`;
      } else svg += `<text x="${x(1) + 6}" y="${y + 4}" style="fill:${cssVar("--muted")}">reference</text>`;
    });
    svg += `</svg>`;
    let contrasts = "";
    if (t.pairs?.length) {
      const CW = 640, ch = 30, CH = t.pairs.length * ch + 30, cl = 300;
      const cx = scaleLog().domain([0.4, 2.5]).range([cl, CW - 20]);
      contrasts = `<h4 style="margin:6px 0">What moved the needle (paired contrasts: ratio a / b on the same windows, log scale)</h4><svg viewBox="0 0 ${CW} ${CH}" width="100%"><line x1="${cx(1)}" x2="${cx(1)}" y1="4" y2="${CH - 22}" stroke="${cssVar("--ink")}" stroke-dasharray="3 3"/>`;
      for (const v of [0.5, 0.75, 1, 1.5, 2]) contrasts += `<text x="${cx(v)}" y="${CH - 6}" text-anchor="middle">${v}</text>`;
      t.pairs.forEach((c, i) => {
        const y = 16 + i * ch, [p, lo, hi] = c.ratio.map((v) => Math.min(2.5, Math.max(0.4, v)));
        const col = c.ratio[2] < 1 ? cssVar("--student") : c.ratio[1] > 1 ? cssVar("--teacher") : cssVar("--baseline");
        contrasts += `<text x="${cl - 8}" y="${y - 2}" text-anchor="end" style="fill:${cssVar("--ink")}">${c.question}</text><text x="${cl - 8}" y="${y + 10}" text-anchor="end">${c.a} vs ${c.b}</text>` +
          `<line x1="${cx(lo)}" x2="${cx(hi)}" y1="${y}" y2="${y}" stroke="${col}" stroke-width="3"/><circle cx="${cx(p)}" cy="${y}" r="5" fill="${col}"/><text x="${cx(hi) + 6}" y="${y + 4}">${c.ratio[0].toFixed(2)}</text>`;
      });
      contrasts += `</svg><p class="small muted">Left of 1: <em>a</em> is better. <span style="color:${cssVar("--student")}">Blue</span> / <span style="color:${cssVar("--teacher")}">orange</span>: interval excludes 1 (a better / b better); grey: no detectable difference.</p>`;
    }
    html += `<div class="card" style="margin:12px 0"><h3 style="margin-top:0">${SUITE_NAMES[suite] ?? suite} · ${t.n_windows} windows</h3>${t.flag ? `<p class="warn">${t.flag}</p>` : ""}${contrasts}
      <p class="small muted">Paired ratio vs the reference arm (log scale). <span style="color:${cssVar("--student")}">Blue</span>: significantly better, <span style="color:${cssVar("--teacher")}">orange</span>: significantly worse, grey: no detectable difference.</p>${svg}
      <div class="table-wrap"><table><thead><tr><th>arm</th><th class="num">rel. WQL</th><th class="num">95% CI</th><th class="num">vs reference</th><th class="num">80% cov.</th></tr></thead><tbody>${arms.map(([m, r]) =>
        `<tr><td>${m}</td><td class="num">${r.rel_wql.toFixed(3)}</td><td class="num">[${r.rel_wql_ci[0].toFixed(3)}, ${r.rel_wql_ci[1].toFixed(3)}]</td><td class="num">${r.vs_reference ? `${r.vs_reference[0].toFixed(3)} [${r.vs_reference[1].toFixed(3)}, ${r.vs_reference[2].toFixed(3)}]` : "–"}</td><td class="num">${(100 * r.cov80).toFixed(0)}%</td></tr>`).join("")}</tbody></table></div>
      <p class="small muted">Zero-shot foundation models on the same suite: ${teach.map(([m, r]) => `${m} ${r.rel_wql.toFixed(3)}`).join(" · ") || "not benchmarked on this suite yet"}</p></div>`;
  }
}
$("stages").innerHTML = html || `<p class="muted">No stage has finished yet.</p>`;
