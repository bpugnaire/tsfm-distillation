import { scaleLog } from "d3-scale";
import { header } from "../lib/nav";
import { cssVar, loadJSON, ratioWords, type Highlight } from "../lib/data";

interface Row { rel_wql: number; rel_wql_ci: [number, number]; cov80: number; vs_reference?: [number, number, number] }
interface Contrast { a: string; b: string; question: string; ratio: [number, number, number] }
interface SuiteTable { pairs?: Contrast[]; arms: Record<string, Row>; zero_shot_models: Record<string, { rel_wql: number; n_params: number }>; n_windows: number; flag: string | null }
interface Stage { study: string; reference_arm: string; suites: Record<string, SuiteTable> }
header();
const study = await loadJSON<Record<string, Stage>>("study.json");
const $ = (id: string) => document.getElementById(id)!;

const STAGES: Record<string, string> = {
  stage_a: "Stage A · what should the teacher look at?",
  stage_b: "Stage B · which teacher, which loss weight?",
  stage_c_lite: "Stage C-lite · more data, bigger student",
  stage_c2: "Stage C2 · new, contamination-filtered data",
  stage_d_pilot: "Stage D pilot · fine-tuning on your data",
  stage_d_final: "Stage D · fine-tuning on your data (3 seeds)",
  stage_e: "Stage E · fine-tuning on teacher labels of your history",
};


// ---- key findings: one controlled comparison per card ---------------------------------------------------------
const res = await loadJSON<{ highlights?: Highlight[] }>("results.json");
const SUITE_SHORT: Record<string, string> = { shapes: "synthetic shapes", multidomain_test: "multi-domain test", fresh_fr_power: "fresh electricity", pedestrian_test: "pedestrian counts" };
function needle(r: [number, number, number]): string {
  const W = 300, H = 34, x = scaleLog().domain([0.25, 8]).range([10, W - 10]).clamp(true);
  const c = r[2] < 1 ? cssVar("--student") : r[1] > 1 ? cssVar("--teacher") : cssVar("--baseline");
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px"><line x1="${x(1)}" x2="${x(1)}" y1="2" y2="${H - 12}" stroke="${cssVar("--ink")}" stroke-dasharray="3 3"/>`;
  for (const t of [0.25, 0.5, 1, 2, 4, 8]) s += `<text x="${x(t)}" y="${H - 1}" text-anchor="middle" style="font-size:10px">${t}</text>`;
  s += `<line x1="${x(r[1])}" x2="${x(r[2])}" y1="11" y2="11" stroke="${c}" stroke-width="4" stroke-linecap="round"/><circle cx="${x(r[0])}" cy="11" r="6" fill="${c}" stroke="${cssVar("--surface")}" stroke-width="2"/></svg>`;
  return s;
}
$("findings").innerHTML = (res.highlights ?? []).map((h) => {
  const sig = h.ratio[2] < 1 || h.ratio[1] > 1;
  return `<div class="finding"><h3>${h.title}</h3><div class="big">${ratioWords(h.ratio[0])}</div><div class="vs">${h.versus}${sig ? "" : " · within noise"}</div>
    ${needle(h.ratio)}<p>${h.text}</p><div class="src">${SUITE_SHORT[h.suite] ?? h.suite} · <a href="#${h.stage}" title="${h.a} vs ${h.b}, ratio ${h.ratio[0].toFixed(2)} [${h.ratio[1].toFixed(2)}, ${h.ratio[2].toFixed(2)}]">details</a></div></div>`;
}).join("");

const SUITE_NAMES: Record<string, string> = { shapes: "Shapes (synthetic, 200 series)", multidomain_test: "Held-out multi-domain test", fresh_fr_power: "Fresh French electricity (zero-shot for everyone)" };
let html = "";
const order = Object.keys(STAGES);
for (const [name, st] of Object.entries(study).sort(([a], [b]) => (order.indexOf(a) + 99) % 99 - (order.indexOf(b) + 99) % 99)) {
  const nArms = Object.keys(Object.values(st.suites)[0]?.arms ?? {}).length;
  html += `<details class="stage" id="${name}"><summary><b>${STAGES[name] ?? name}</b><span class="q">${nArms} variants, ${Object.keys(st.suites).length} test sets</span></summary><div class="body"><p class="small muted">Arms with a <code>_s0/_s1/_s2</code> suffix are seeds; the others are one seed (exploratory): differences smaller than seed-to-seed noise need confirmation.</p>`;
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
  html += "<!--/stage-->";
}
$("stages").innerHTML = html.replaceAll("<!--/stage-->", "</div></details>") || `<p class="muted">No stage has finished yet.</p>`;

// open a collapsed stage when a link points at it
function openFromHash(): void {
  const el = location.hash ? document.getElementById(location.hash.slice(1)) : null;
  if (el instanceof HTMLDetailsElement) { el.open = true; el.scrollIntoView({ block: "start" }); }
}
addEventListener("hashchange", openFromHash);
openFromHash();
