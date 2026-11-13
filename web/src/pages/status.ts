import { header } from "../lib/nav";
import { cssVar, loadJSON, type Results } from "../lib/data";

interface Item { state: "done" | "todo" | "superseded"; text: string; section: string }
interface Status {
  current_focus: string;
  phases: { title: string; items: Item[] }[];
  pending_decisions: { id: string; question: string; options?: string[]; note?: string }[];
  decided: { date: string; what: string; ref: string }[];
  journal: { title: string; next: string }[];
  models: { file: string; size_mb: number }[];
}
header();
const [st, res] = await Promise.all([loadJSON<Status>("status.json"), loadJSON<Results>("results.json")]);
const $ = (id: string) => document.getElementById(id)!;
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");

$("focus").innerHTML = `<strong>Current focus:</strong> ${esc(st.current_focus)}`;

const all = st.phases.flatMap((p) => p.items);
const done = all.filter((i) => i.state === "done").length;
const fresh = res.suites.fresh_fr_power.rows;
const best = fresh.find((r) => r.role !== "student" && r.model !== "seasonal_naive")!;
const student = fresh.find((r) => r.role === "student");
$("kpis").innerHTML = [
  [`${done} / ${all.length}`, "plan items done"],
  [`${st.pending_decisions.length}`, "decisions waiting for you"],
  [`${best.rel_wql.toFixed(3)}`, `best on fresh data (${best.model})`],
  [student ? student.rel_wql.toFixed(3) : "TBD", "student on fresh data (preview, not distilled)"],
  ["TBD", "distilled student"],
].map(([v, l]) => `<div class="card kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");

$("phases").innerHTML = st.phases.map((p, k) => {
  const n = p.items.length, d = p.items.filter((i) => i.state === "done").length, s = p.items.filter((i) => i.state === "superseded").length;
  let sec = "";
  const items = p.items.map((i) => {
    const head = i.section && i.section !== sec ? `<li class="sec" style="padding-left:0">${esc((sec = i.section))}</li>` : "";
    return `${head}<li class="${i.state}">${esc(i.text)}</li>`;
  }).join("");
  return `<details class="phase"${k === 3 || k === 4 ? " open" : ""}><summary><span>${esc(p.title)}</span>
    <span class="bar" title="${d} done, ${s} superseded, ${n - d - s} to do"><span style="width:${(100 * d) / n}%;background:${cssVar("--student")}"></span><span style="width:${(100 * s) / n}%;background:${cssVar("--teacher")}"></span></span>
    <span class="small muted" style="text-align:right;font-variant-numeric:tabular-nums">${d}/${n}</span></summary><ul class="items">${items}</ul></details>`;
}).join("");

$("pending").innerHTML = st.pending_decisions.map((d) => `<div class="decision"><div class="q">${esc(d.question)}</div>
  ${d.options ? `<div class="small muted">Options: ${d.options.map(esc).join(" · ")}</div>` : ""}${d.note ? `<div class="small" style="color:var(--ink-2)">${esc(d.note)}</div>` : ""}</div>`).join("");
$("decided").innerHTML = `<table><tbody>${st.decided.map((d) => `<tr><td class="small muted" style="white-space:nowrap">${d.date}</td><td class="small">${esc(d.what)}<br><code class="muted">${esc(d.ref)}</code></td></tr>`).join("")}</tbody></table>`;
$("journal").innerHTML = st.journal.slice(-3).reverse().map((j) => `<p style="margin-top:0"><b>${esc(j.title)}</b><br><span class="muted">Next:</span> ${esc(j.next)}</p>`).join("");
$("models").innerHTML = `<table><tbody>${st.models.map((m) => `<tr><td><code>${m.file}</code></td><td class="num">${m.size_mb.toFixed(2)} MB</td></tr>`).join("")}</tbody></table>`;
