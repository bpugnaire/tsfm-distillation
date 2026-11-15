import { header } from "../lib/nav";
import { fmtParams, isPreview, loadJSON, roleColor, type Gallery, type GalleryModel, type Results } from "../lib/data";
import { drawFan } from "../lib/fan";

header();
const [g, res] = await Promise.all([loadJSON<Gallery>("gallery.json"), loadJSON<Results>("results.json")]);
const shapesSummary = res.suites.shapes?.rows ?? [];
const relWql = new Map(shapesSummary.map((r) => [r.model, r.rel_wql]));
const visible = g.models.filter((m) => !m.name.endsWith("-onnx-fp32")); // identical to the torch student
let order: GalleryModel[] = [...visible];

document.getElementById("legend")!.innerHTML = (["baseline", "comparator", "teacher", "student"] as const)
  .map((r) => `<span><span class="swatch" style="background:${roleColor(r)}"></span>${r}</span>`).join(" &nbsp; ");
document.getElementById("foot")!.textContent =
  `Cells show seed 0 of each shape. "accuracy" sorts by relative WQL over 10 seeds per shape (Results page). ` +
  `Latency: CPU, 1 thread, batch 1, p50, on ${g.hardware.cpu}. Toto 2.0 tails (1%/99%) are extrapolated from its 0.1–0.9 output.`;

function render(): void {
  const grid = document.getElementById("grid")!;
  grid.innerHTML = "";
  grid.style.gridTemplateColumns = `110px repeat(${order.length}, minmax(120px, 1fr))`;
  grid.append(Object.assign(document.createElement("div"), { className: "hd" }));
  for (const m of order) {
    const d = document.createElement("div");
    d.className = "hd";
    const rel = relWql.get(m.name);
    const nice: Record<string, string> = { "student-distilled": "our student · distilled", "student-gt-only": "our student · no teacher" };
    if (nice[m.name]) d.style.cssText = "background:var(--accent-wash);border-radius:6px";
    d.innerHTML = `<b><span class="swatch" style="background:${roleColor(m.role)}"></span>${nice[m.name] ?? m.name}</b>` +
      `${fmtParams(m.n_params)} · ${m.latency_ms ?? "–"} ms` + (rel ? ` · rel ${rel.toFixed(2)}` : "") +
      (isPreview(m.name) ? `<br><span class="tag">preview</span>` : "");
    grid.append(d);
  }
  for (const s of g.shapes) {
    const lab = document.createElement("div");
    lab.className = "rowlabel";
    lab.textContent = s.name.replaceAll("_", " ");
    lab.title = s.description;
    grid.append(lab);
    const wqls = order.map((m) => g.cells[m.name]?.[s.name]?.wql ?? Infinity);
    const best = Math.min(...wqls);
    order.forEach((m, j) => {
      const c = g.cells[m.name]?.[s.name];
      const cell = document.createElement("div");
      cell.className = "cell" + (wqls[j] === best ? " best" : "");
      cell.title = `${m.name} on ${s.name}: ${s.description}`;
      cell.onclick = () => { location.href = `index.html?shape=${s.name}`; };
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      cell.append(svg);
      const meta = document.createElement("div");
      meta.className = "m";
      meta.textContent = c?.wql != null ? `WQL ${c.wql.toFixed(3)}` : "–";
      cell.append(meta);
      grid.append(cell);
      if (c) drawFan(svg, { context: s.context, future: s.future, q: c.q, levels: g.levels, color: roleColor(m.role), width: 160, height: 70, showContext: 128 });
    });
  }
}

function setSort(kind: "default" | "acc" | "speed"): void {
  for (const k of ["default", "acc", "speed"]) document.getElementById(`s-${k}`)!.setAttribute("aria-pressed", String(k === kind));
  order = [...visible];
  if (kind === "acc") order.sort((a, b) => (relWql.get(a.name) ?? 9) - (relWql.get(b.name) ?? 9));
  if (kind === "speed") order.sort((a, b) => (a.latency_ms ?? 1e9) - (b.latency_ms ?? 1e9));
  render();
}
document.getElementById("s-default")!.onclick = () => setSort("default");
document.getElementById("s-acc")!.onclick = () => setSort("acc");
document.getElementById("s-speed")!.onclick = () => setSort("speed");
render();
