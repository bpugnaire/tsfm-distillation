// Script of the two explainer posts: post-data.html (training data) and post-student.html (architecture + one inference).
// All data comes from public/data/explain.json, written by scripts/build_explainers.py.
import { scaleLinear, type ScaleLinear } from "d3-scale";
import { select, type Selection } from "d3-selection";
import { area, line } from "d3-shape";
import { cssVar, hideTip, loadJSON, showTip, type Snippet } from "../lib/data";
import { header } from "../lib/nav";
import { highlightStatic, renderCode, renderMath } from "../lib/rich";

type N = number | null;
interface Example { source: "real" | "synth"; name: string; row: number; window_id: string; context: N[]; future: N[]; teacher: number[][]; n_missing: number }
interface Group { group: string; role: "range" | "knowledge" | "plumbing"; layer: number | null; params: number; shapes: [string, number[]][] }
interface Layer { query_attention: number[][]; attention_mean: number[][]; norm_in: number[]; attn_update: number[]; ffn_update: number[] }
interface Explain {
  run: string;
  teacher: string;
  model: {
    config: { context_length: number; horizon: number; patch_size: number; d_model: number; n_layers: number; n_heads: number; ff_mult: number; quantile_levels: number[]; n_patches: number };
    n_params: number;
    groups: Group[];
    trace: {
      series_id: string; target_start: string; context: number[]; future: number[]; loc: number; scale: number; z: number[];
      patch_inputs: number[][]; embeddings: number[][]; query: number[]; pos_norm: number[]; emb_norm: number[]; layers: Layer[];
      final: number[]; head_raw: number[][]; z_quantiles: number[][]; quantiles: number[][]; i_med: number;
      scale_invariance_rel_err: number; invariance_transform: [number, number];
    };
  };
  data: {
    corpus_key: string; counts: Record<string, number>; context_length: number; horizon: number;
    datasets: { dataset: string; domain: string; windows: number; series: number }[];
    n_pool_series: number;
    contamination: { candidates: number; kept: number; dropped_by_dataset: Record<string, number> };
    series_example: { dataset: string; series_id: string; length: number; bin: number; values: N[]; splits: Record<string, [number, number]>;
      origins: number[]; grid: number[]; zoom: { start: number; values: N[]; slots: number[]; sampled: boolean[] } };
    synthetic: { kinds: Record<string, number>; shapes: Record<string, number> };
    examples: Example[];
    cache: { path: string; n_shards: number; shard_size: number; n_windows: number; bytes: number; manifest: Record<string, unknown>; first_shards: Record<string, number> };
    validation: Record<string, number>;
    rte: { daily_start: string; daily: number[]; train_end: string; val_start: string; val_end: string; fresh_start: string; last: string; n_fresh_windows_france: number; series: string[] };
  };
}

header();
renderMath();
highlightStatic();
const $ = (id: string) => document.getElementById(id);
const E = await loadJSON<Explain>("explain.json");
const figs = document.querySelectorAll<HTMLElement>("figure.code[data-ref]");
if (figs.length) {
  const code = await loadJSON<Record<string, Snippet>>("code.json");
  for (const f of figs) { const c = code[f.dataset.ref!]; if (c) renderCode(f, c); }
}

// ---------------------------------------------------------------- helpers
type Svg = Selection<SVGSVGElement, unknown, HTMLElement, unknown>;
const fmt = (n: number) => n.toLocaleString("en-US");
const mix = (v: string, pct: number) => `color-mix(in srgb, var(${v}) ${pct}%, var(--surface))`;
function svgOf(id: string, W: number, H: number): Svg {
  const s = select<SVGSVGElement, unknown>(`#${id}`).attr("viewBox", `0 0 ${W} ${H}`).attr("width", "100%");
  s.selectAll("*").remove();
  return s as unknown as Svg;
}
const ROLE_VAR: Record<string, string> = { range: "--range", knowledge: "--knowledge", plumbing: "--fixed", fixed: "--fixed", flow: "--flow" };
function diverging(v: number, max: number): string {
  const t = Math.min(1, Math.abs(v) / (max || 1));
  return v >= 0 ? mix("--pos", Math.round(8 + 85 * t)) : mix("--flow", Math.round(8 + 85 * t));
}
function pathOf(xs: (i: number) => number, y: (v: number) => number, vals: N[]): string {
  return line<N>().defined((d) => d != null).x((_, i) => xs(i)).y((d) => y(d as number))(vals) ?? "";
}
function extent(...arrs: N[][]): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (const a of arrs) for (const v of a) if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  return [lo, hi];
}
function yAxis(svg: Svg, y: ScaleLinear<number, number>, x0: number, x1: number, n = 4, f = (v: number) => fmt(v)): void {
  for (const t of y.ticks(n)) {
    svg.append("line").attr("x1", x0).attr("x2", x1).attr("y1", y(t)).attr("y2", y(t)).attr("stroke", cssVar("--grid"));
    svg.append("text").attr("x", x0 - 6).attr("y", y(t) + 4).attr("text-anchor", "end").text(f(t));
  }
}
function fan(svg: Svg, q: number[][], x: (i: number) => number, y: (v: number) => number, colour: string, iMed: number): void {
  const K = q.length;
  for (const [lo, hi, op] of [[2, K - 3, 0.22], [4, K - 5, 0.38]] as const)
    svg.append("path").attr("d", area<number>().x((_, i) => x(i)).y0((d) => y(d)).y1((_, i) => y(q[hi][i]))(q[lo]) ?? "").attr("fill", colour).attr("opacity", op);
  svg.append("path").attr("d", line<number>().x((_, i) => x(i)).y((d) => y(d))(q[iMed]) ?? "").attr("fill", "none").attr("stroke", colour).attr("stroke-width", 2);
}
function bracket(svg: Svg, x0: number, x1: number, y: number, label: string, colour: string): void {
  svg.append("path").attr("d", `M${x0},${y + 5} V${y} H${x1} V${y + 5}`).attr("fill", "none").attr("stroke", colour).attr("stroke-width", 1.5);
  svg.append("text").attr("x", (x0 + x1) / 2).attr("y", y - 5).attr("text-anchor", "middle").style("fill", colour).style("font-weight", "600").text(label);
}
function tip(el: Element, html: string): void {
  el.addEventListener("mousemove", (e) => showTip(e as MouseEvent, html));
  el.addEventListener("mouseleave", hideTip);
}

// ================================================================ post-data.html
if ($("d-anatomy")) {
  const D = E.data, T = D.context_length, H = D.horizon;
  const nReal = D.counts.multidomain_v2, nSyn = D.counts.synth;
  $("d-numbers")!.innerHTML = [
    [fmt(nReal + nSyn), "training windows"],
    [fmt(nReal), `real windows, from ${fmt(D.n_pool_series)} series in ${D.datasets.length} datasets`],
    [fmt(nSyn), "synthetic windows, regenerated from seeds"],
    [`${(D.cache.bytes / 1e9).toFixed(2)} GB`, `of teacher labels, in ${D.cache.n_shards} shards`],
    ["0", "windows of French electricity: that is the test"],
  ].map(([b, s]) => `<div><b>${b}</b><span>${s}</span></div>`).join("");

  // ---- 1. anatomy of one window
  const KIND: Record<string, string> = { kernel: "synthetic: Gaussian process (KernelSynth)", shape: "synthetic: random shape", mix: "synthetic: mix of several (TSMixup)" };
  const NAME: Record<string, string> = { PEMS03: "road traffic (PEMS03)", london_smart_meters_with_missing: "smart meter (London)", beijing_air_quality: "air quality (Beijing)" };
  let ex = 0;
  const exTabs = () => {
    $("d-ex-tabs")!.innerHTML = D.examples.map((e, i) => `<button type="button" data-i="${i}" aria-pressed="${i === ex}">${e.source === "real" ? NAME[e.name] ?? e.name : KIND[e.name]}</button>`).join("");
    $("d-ex-tabs")!.querySelectorAll<HTMLButtonElement>("button").forEach((b) => (b.onclick = () => { ex = +b.dataset.i!; exTabs(); anatomy(); }));
  };
  const anatomy = () => {
    const e = D.examples[ex], W = 680, Hh = 270, m = { l: 52, r: 10, t: 34, b: 28 };
    const svg = svgOf("d-anatomy", W, Hh);
    const x = scaleLinear().domain([0, T + H]).range([m.l, W - m.r]);
    const [lo, hi] = extent(e.context, e.future, e.teacher[0], e.teacher[12]);
    const y = scaleLinear().domain([lo, hi]).nice().range([Hh - m.b, m.t + 8]);
    yAxis(svg, y, m.l, W - m.r, 4, (v) => (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(0)}k` : `${+v.toFixed(2)}`));
    svg.append("rect").attr("x", x(T)).attr("y", m.t).attr("width", x(T + H) - x(T)).attr("height", Hh - m.b - m.t).style("fill", mix("--teacher", 7));
    svg.append("line").attr("x1", x(T)).attr("x2", x(T)).attr("y1", m.t).attr("y2", Hh - m.b).attr("stroke", cssVar("--axis")).attr("stroke-dasharray", "4 3");
    svg.append("path").attr("d", pathOf((i) => x(i), y, e.context)).attr("fill", "none").attr("stroke", cssVar("--ink-2")).attr("stroke-width", 1.2);
    fan(svg, e.teacher, (i) => x(T + i), y, cssVar("--teacher"), 6);
    svg.append("path").attr("d", pathOf((i) => x(T + i), y, e.future)).attr("fill", "none").attr("stroke", cssVar("--ink")).attr("stroke-width", 1.4);
    bracket(svg, x(0), x(T) - 4, m.t - 6, `context: ${T} steps (what the student reads)`, cssVar("--ink-2"));
    bracket(svg, x(T) + 2, x(T + H), m.t - 6, `target: ${H}`, cssVar("--teacher"));
    for (const t of [0, 128, 256, 384, 512, 576]) svg.append("text").attr("x", x(t)).attr("y", Hh - m.b + 16).attr("text-anchor", "middle").text(t === T ? "origin" : t < T ? `−${T - t}` : `+${t - T}`);
    $("d-ex-cap")!.innerHTML = `Corpus row ${fmt(e.row)}, window id <code>${e.window_id}</code>. ${e.source === "synth"
      ? "Its real future comes from the generator, which knows the exact continuation."
      : `Missing steps in the context: ${e.n_missing}.`} The teacher's 13 quantiles come straight from the cache shard; training uses only its 9 native levels (0.1 to 0.9).`;
  };
  exTabs();
  anatomy();

  // ---- 2. composition
  const DOMAINS = [...new Set(D.datasets.map((d) => d.domain))];
  const byDomain = DOMAINS.map((dm) => ({ dm, n: D.datasets.filter((d) => d.domain === dm).reduce((s, d) => s + d.windows, 0) })).sort((a, b) => b.n - a.n);
  $("d-nseries")!.textContent = fmt(D.n_pool_series);
  $("d-nds")!.textContent = `${D.datasets.length}`;
  {
    const W = 680, Hh = 150, m = { l: 80, r: 10 }, svg = svgOf("d-mix", W, Hh);
    const x = scaleLinear().domain([0, nReal]).range([m.l, W - m.r]);
    const rows: [string, { label: string; n: number }[], string][] = [
      ["real", byDomain.map((d) => ({ label: d.dm, n: d.n })), "--ink-2"],
      ["synthetic", [{ label: "Gaussian process", n: D.synthetic.kinds.kernel }, { label: "random shapes", n: D.synthetic.kinds.shape }, { label: "mixes", n: D.synthetic.kinds.mix }], "--fixed"],
    ];
    rows.forEach(([name, segs, v], r) => {
      const y0 = 14 + r * 70;
      svg.append("text").attr("x", m.l - 10).attr("y", y0 + 22).attr("text-anchor", "end").attr("class", "lbl-strong").text(name);
      let acc = 0;
      segs.forEach((s, i) => {
        const g = svg.append("g");
        g.append("rect").attr("x", x(acc) + 0.5).attr("y", y0).attr("width", Math.max(0, x(acc + s.n) - x(acc) - 1)).attr("height", 34).attr("rx", 3)
          .style("fill", mix(v, 85 - (i % 6) * 13));
        if (x(acc + s.n) - x(acc) > 54) g.append("text").attr("x", x(acc) + 6).attr("y", y0 + 52).attr("class", "lbl").style("font-size", "11.5px").text(s.label);
        tip(g.node()!, `<b>${s.label}</b>: ${fmt(s.n)} windows (${((100 * s.n) / (name === "real" ? nReal : nSyn)).toFixed(1)}%)`);
        acc += s.n;
      });
    });
  }
  {
    const ds = D.datasets, W = 680, rowH = 15, m = { l: 210, r: 60, t: 6, b: 20 }, Hh = m.t + m.b + ds.length * rowH;
    const svg = svgOf("d-datasets", W, Hh), cap = Math.max(...ds.map((d) => d.windows));
    const x = scaleLinear().domain([0, cap]).range([m.l, W - m.r]);
    ds.forEach((d, i) => {
      const y = m.t + i * rowH, g = svg.append("g");
      g.append("rect").attr("x", m.l).attr("y", y + 2).attr("width", x(d.windows) - m.l).attr("height", rowH - 4).attr("rx", 2).style("fill", mix("--ink-2", d.windows === cap || d.windows === cap - 1 ? 60 : 35));
      g.append("text").attr("x", m.l - 8).attr("y", y + rowH - 4).attr("text-anchor", "end").attr("class", "lbl").style("font-size", "11.5px").text(`${d.dataset} · ${d.domain}`);
      g.append("text").attr("x", x(d.windows) + 5).attr("y", y + rowH - 4).style("font-size", "11px").text(fmt(d.windows));
      tip(g.node()!, `<b>${d.dataset}</b> (${d.domain}): ${fmt(d.windows)} windows from ${fmt(d.series)} series`);
    });
    svg.append("line").attr("x1", x(cap)).attr("x2", x(cap)).attr("y1", 0).attr("y2", Hh - m.b).attr("stroke", cssVar("--axis")).attr("stroke-dasharray", "4 3");
    svg.append("text").attr("x", x(cap)).attr("y", Hh - 6).attr("text-anchor", "middle").text(`equal share: ${fmt(cap)}`);
  }

  // ---- 3. one series cut into windows
  const S = D.series_example;
  $("d-ser-name")!.innerHTML = `<code>${S.dataset}</code>, series ${S.series_id}, ${fmt(S.length)} hours`;
  {
    const W = 680, Hh = 210, m = { l: 14, r: 14, t: 26, b: 58 }, svg = svgOf("d-series", W, Hh);
    const x = scaleLinear().domain([0, S.length]).range([m.l, W - m.r]);
    const [lo, hi] = extent(S.values);
    const y = scaleLinear().domain([lo, hi]).range([Hh - m.b, m.t]);
    const SPL: [string, string, string][] = [["train", "--s-train", "train · first 70%"], ["val", "--s-val", "val · 15%"], ["test", "--s-test", "test · 15%"]];
    for (const [k, v, lab] of SPL) {
      const [a, b] = S.splits[k];
      svg.append("rect").attr("x", x(a)).attr("y", m.t).attr("width", x(b) - x(a)).attr("height", Hh - m.b - m.t).style("fill", mix(v, 10));
      svg.append("text").attr("x", x(a) + 4).attr("y", m.t - 8).style("fill", cssVar(v)).style("font-weight", "600").text(lab);
    }
    svg.append("path").attr("d", pathOf((i) => x(i * S.bin), y, S.values)).attr("fill", "none").attr("stroke", cssVar("--ink-2")).attr("stroke-width", 1);
    const yt = Hh - m.b + 10;
    for (const o of S.grid) svg.append("line").attr("x1", x(o)).attr("x2", x(o)).attr("y1", yt).attr("y2", yt + 8).attr("stroke", cssVar("--axis"));
    for (const o of S.origins) svg.append("line").attr("x1", x(o)).attr("x2", x(o)).attr("y1", yt).attr("y2", yt + 16).attr("stroke", cssVar("--s-train")).attr("stroke-width", 1.5);
    svg.append("text").attr("x", m.l).attr("y", Hh - 8).attr("class", "lbl").style("font-size", "12px")
      .text(`ticks: the ${S.grid.length} grid slots in the train part (grey) and the ${S.origins.length} the corpus drew (blue)`);
    const z0 = S.zoom.start, z1 = S.zoom.start + S.zoom.values.length;
    svg.append("rect").attr("x", x(z0)).attr("y", m.t).attr("width", Math.max(2, x(z1) - x(z0))).attr("height", Hh - m.b - m.t).attr("fill", "none").attr("stroke", cssVar("--ink")).attr("stroke-width", 1.2);
    $("d-series-cap")!.textContent = `Bin means of ${S.bin} hours. The black box is the stretch zoomed in below. Validation and test windows of this series come from the other two bands, and only from the multi-domain test suite: the student's checkpoint is chosen on a separate validation set (${fmt(D.validation.multidomain)} windows).`;
  }
  {
    const Z = S.zoom, W = 680, Hh = 250, m = { l: 14, r: 14, t: 10, b: 24 }, svg = svgOf("d-zoom", W, Hh);
    const x = scaleLinear().domain([Z.start, Z.start + Z.values.length]).range([m.l, W - m.r]);
    const [lo, hi] = extent(Z.values);
    const y = scaleLinear().domain([lo, hi]).range([100, m.t]);
    svg.append("path").attr("d", pathOf((i) => x(Z.start + i), y, Z.values)).attr("fill", "none").attr("stroke", cssVar("--ink-2")).attr("stroke-width", 1.2);
    Z.slots.forEach((o, k) => {
      const yy = 120 + k * 36, on = Z.sampled[k];
      svg.append("rect").attr("x", x(o - T)).attr("y", yy).attr("width", x(o) - x(o - T)).attr("height", 22).attr("rx", 3).style("fill", mix("--ink-2", on ? 30 : 10));
      svg.append("rect").attr("x", x(o)).attr("y", yy).attr("width", x(o + H) - x(o)).attr("height", 22).attr("rx", 3).style("fill", on ? cssVar("--s-train") : mix("--s-train", 25));
      svg.append("text").attr("x", x(o - T) + 6).attr("y", yy + 15).attr("class", "lbl").style("font-size", "12px").text(`slot ${k + 1}: context ${T}${on ? "" : " · not drawn"}`);
      svg.append("line").attr("x1", x(o)).attr("x2", x(o)).attr("y1", y(hi)).attr("y2", yy + 22).attr("stroke", cssVar("--s-train")).attr("stroke-dasharray", "2 3").attr("opacity", 0.6);
    });
    svg.append("text").attr("x", W - m.r).attr("y", Hh - 4).attr("text-anchor", "end").text(`stride ${H}: targets tile, contexts overlap by ${T - H} steps`);
  }

  // ---- 4. synthetic kinds
  const K = D.synthetic.kinds;
  $("d-kinds")!.innerHTML = [
    `<b>${fmt(K.kernel)} Gaussian-process series</b> (KernelSynth, from Chronos): random sums and products of periodic, smooth, linear and noise kernels. They teach seasonality and trends.`,
    `<b>${fmt(K.shape)} random shapes</b>: the ${Object.keys(D.synthetic.shapes).length} shape families of our test gallery (level shifts, spikes, gaps, chirps…) with randomised parameters and training-only seeds, about ${fmt(Math.round(K.shape / Object.keys(D.synthetic.shapes).length))} each.`,
    `<b>${fmt(K.mix)} mixes</b> of 2–3 of the above with random weights (TSMixup).`,
  ].map((s) => `<li>${s}</li>`).join("");

  // ---- 5. contamination + RTE timeline
  const dropped = Object.entries(D.contamination.dropped_by_dataset).sort((a, b) => b[1] - a[1]);
  $("d-contam")!.innerHTML = `Result: ${fmt(D.contamination.candidates)} candidate series, <b>${fmt(D.contamination.candidates - D.contamination.kept)} dropped</b>, ${fmt(D.contamination.kept)} kept. Most drops: ${dropped.slice(0, 4).map(([k, v]) => `<code>${k}</code> (${v})`).join(", ")}.`;
  {
    const R = D.rte, W = 680, Hh = 200, m = { l: 48, r: 10, t: 28, b: 26 }, svg = svgOf("d-rte", W, Hh);
    const day = (s: string) => (Date.parse(s) - Date.parse(R.daily_start)) / 864e5;
    const x = scaleLinear().domain([0, R.daily.length]).range([m.l, W - m.r]);
    const [lo, hi] = extent(R.daily);
    const y = scaleLinear().domain([lo, hi]).nice().range([Hh - m.b, m.t]);
    yAxis(svg, y, m.l, W - m.r, 3, (v) => `${(v / 1000).toFixed(0)} GW`);
    const bands: [number, number, string, string][] = [
      [0, day(R.train_end), "--s-train", `train ≤ ${R.train_end} (fine-tuning studies only)`],
      [day(R.val_start), day(R.val_end), "--s-val", "val"],
      [day(R.fresh_start), R.daily.length, "--s-test", "fresh test"],
    ];
    bands.forEach(([a, b, v, lab], i) => {
      svg.append("rect").attr("x", x(a)).attr("y", m.t).attr("width", Math.max(2, x(b) - x(a))).attr("height", Hh - m.b - m.t).style("fill", mix(v, i === 0 ? 8 : 30));
      svg.append("text").attr("x", i === 0 ? x(a) + 4 : x(b)).attr("y", m.t - 8 - (i === 1 ? 0 : 0)).attr("text-anchor", i === 0 ? "start" : "end").style("fill", cssVar(v)).style("font-weight", "600").style("font-size", "11.5px").text(i === 1 ? "" : lab);
    });
    svg.append("path").attr("d", pathOf((i) => x(i), y, R.daily)).attr("fill", "none").attr("stroke", cssVar("--ink-2")).attr("stroke-width", 0.8);
    for (const yr of [2014, 2018, 2022, 2026]) svg.append("text").attr("x", x(day(`${yr}-01-01`))).attr("y", Hh - 8).attr("text-anchor", "middle").text(yr);
    $("d-rte-cap")!.innerHTML = `National demand, daily means, ${R.daily_start} to ${R.last} (RTE éCO2mix, ${R.series.length} series: France and 12 regions).
      <span class="c-val">Validation</span>: ${R.val_start} to ${R.val_end}. <span class="c-test">Fresh test</span>: targets from ${R.fresh_start}, the day after the last model release. The gap in between is unused.
      The <span class="c-train">train</span> period is used only by the fine-tuning experiments and the ground-truth baseline student, never by distillation.`;
  }

  // ---- 6. disk
  const C = D.cache, per = C.bytes / C.n_windows;
  $("d-tree")!.innerHTML = [
    `data/processed/`,
    `└── corpus_train_${D.corpus_key}.npz        <span class="dim"># ids, source, context (${fmt(nReal + nSyn)} × ${T} float32), future</span>`,
    `${C.path}`,
    `├── manifest.json                     <span class="dim"># teacher, checkpoint, package, levels, git SHA, ids hash</span>`,
    `├── shard_00000.parquet               <span class="dim"># ${fmt(C.shard_size)} rows: window_id (string), q (binary)</span>`,
    `├── …`,
    `└── shard_${String(C.n_shards - 1).padStart(5, "0")}.parquet               <span class="dim"># ${C.n_shards} shards, ${(C.bytes / 1e9).toFixed(2)} GB, ${fmt(Math.round(per))} bytes per window</span>`,
  ].join("\n");
  const K13 = (C.manifest.levels as number[]).length, H64 = C.manifest.horizon as number;
  $("d-blob-bytes")!.textContent = `${8 + 2 * K13 * H64} bytes`;
  {
    const W = 680, Hh = 76, svg = svgOf("d-blob", W, Hh), total = 8 + 2 * K13 * H64;
    const x = scaleLinear().domain([0, total]).range([0, W]);
    const segs: [number, number, string, string][] = [[0, 4, "--fixed", "loc f32"], [4, 8, "--fixed", "scale f32"], [8, total, "--teacher", `${K13} × ${H64} float16 of (q − loc) / scale, in [−1, 1]`]];
    segs.forEach(([a, b, v, lab], i) => {
      svg.append("rect").attr("x", x(a) + 0.5).attr("y", 8).attr("width", Math.max(14, x(b) - x(a) - 1)).attr("height", 34).attr("rx", 3).style("fill", mix(v, i === 2 ? 40 : 55));
      svg.append("text").attr("x", i === 2 ? x(a) + 40 : i === 0 ? 0 : 40).attr("y", 62).style("font-size", "11.5px").attr("class", "lbl").text(i === 2 ? lab : i === 0 ? "loc, scale: 8 bytes" : "");
    });
    svg.append("text").attr("x", W).attr("y", 62).attr("text-anchor", "end").style("font-size", "11.5px").text(`${2 * K13 * H64} bytes`);
  }
  $("d-manifest")!.textContent = JSON.stringify({ ...C.manifest, shards: { ...C.first_shards, "…": `${C.n_shards - 2} more` } }, null, 1);
}

// ================================================================ post-student.html
if ($("m-diagram")) {
  const M = E.model, c = M.config, tr = M.trace, T = c.context_length, H = c.horizon, P = c.patch_size, NP = c.n_patches, d = c.d_model, K = c.quantile_levels.length;
  const sum = (f: (g: Group) => boolean) => M.groups.filter(f).reduce((s, g) => s + g.params, 0);
  const P_ATT = sum((g) => g.group === "attention"), P_FFN = sum((g) => g.group === "feed-forward"), P_NORM = sum((g) => g.group === "layer norms");
  const P_EMB = sum((g) => g.group === "patch embedding"), P_POS = sum((g) => g.group === "position table"), P_Q = sum((g) => g.group === "query token"), P_HEAD = sum((g) => g.group === "quantile head");
  const pct = (n: number) => `${((100 * n) / M.n_params).toFixed(n / M.n_params < 0.01 ? 2 : 0)}%`;

  // ---- 1. diagram
  interface Block { id: string; role: keyof typeof ROLE_VAR; title: string; shape: string; params: number; text: string; h?: number }
  const blocks: Block[] = [
    { id: "in", role: "flow", title: "Raw input", shape: `(1, ${T})`, params: 0, text: `${T} raw values, any unit; <code>NaN</code> marks a missing step. Shorter inputs are left-padded with <code>NaN</code>; longer ones are cut to the last ${T} steps.` },
    { id: "scaler", role: "fixed", title: "Scaler: median, mean |deviation|, asinh", shape: `(1, ${T})`, params: 0, text: `Computes two numbers from the observed context, <b>loc</b> (median) and <b>scale</b> (mean absolute deviation from it), then z = asinh((x − loc) / scale). No weights. This is why <b>amplitude is unlimited</b>: every series reaches the network at the same size.` },
    { id: "patch", role: "range", title: `Cut into ${NP} patches of ${P}, plus a mask`, shape: `(1, ${NP}, ${2 * P})`, params: 0, text: `A reshape: ${NP} × ${P} = <b>${T}</b>. Each patch is its ${P} values (0 where missing) next to ${P} observed flags. Patch length × number of patches <b>is</b> the context length.` },
    { id: "embed", role: "knowledge", title: `Patch embedding: MLP ${2 * P} → ${2 * d} → ${d}`, shape: `(1, ${NP}, ${d})`, params: P_EMB, text: `Turns each patch into a ${d}-number token. The same small network for every patch: it learns what local shapes (a ramp, a peak, a gap) look like. <b>${fmt(P_EMB)} weights</b>.` },
    { id: "pos", role: "range", title: `+ position table (${NP + 1} × ${d}) and the query token`, shape: `(1, ${NP + 1}, ${d})`, params: P_POS + P_Q, text: `A learned vector per position tells attention <i>where</i> each patch is. The table has <b>${NP + 1} rows</b>: ${NP} patches + 1 query token. This is the hard cap on the input: there is no row for a 33rd patch. The query token (${P_Q} learned numbers) is appended last; it is the slot the forecast will be read from.` },
    { id: "enc", role: "knowledge", title: `${c.n_layers} × transformer layer (pre-norm)`, shape: `(1, ${NP + 1}, ${d})`, params: P_ATT + P_FFN + P_NORM, h: 92, text: `Each layer: LayerNorm → <b>attention</b> with ${c.n_heads} heads (every token reads every other; ${fmt(P_ATT / c.n_layers)} weights) → add; LayerNorm → <b>feed-forward</b> ${d} → ${d * c.ff_mult} → ${d} (${fmt(P_FFN / c.n_layers)} weights) → add. <b>${pct(P_ATT + P_FFN)} of all weights</b>: this is where the knowledge is.` },
    { id: "norm", role: "fixed", title: "Final LayerNorm, keep the query token only", shape: `(1, ${d})`, params: sum((g) => g.group === "layer norms" && g.layer == null), text: `The ${NP} patch tokens are dropped. Everything the head will know is in these ${d} numbers.` },
    { id: "head", role: "range", title: `Head: Linear ${d} → ${K * H} = ${K} × ${H}`, shape: `(1, ${K}, ${H})`, params: P_HEAD, text: `One linear layer outputs <b>${K} quantile levels × ${H} steps</b>. Its size fixes the horizon and the levels: ${d} × ${K * H} + ${K * H} = <b>${fmt(P_HEAD)} weights</b>, ${pct(P_HEAD)} of the model.` },
    { id: "mono", role: "fixed", title: "Median + softplus gaps, cumulated outward", shape: `(1, ${K}, ${H})`, params: 0, text: `The median output is kept; the other 12 outputs become positive gaps (softplus) added outward from it. Quantiles <b>cannot cross</b>, whatever the weights.` },
    { id: "unscale", role: "fixed", title: "Unscale: sinh(z) × scale + loc", shape: `(1, ${K}, ${H})`, params: 0, text: `The inverse of the scaler, with the same loc and scale. The output is in the input's units.` },
    { id: "out", role: "flow", title: "13 quantiles × 64 steps", shape: `(1, ${K}, ${H})`, params: 0, text: `Levels ${c.quantile_levels.join(", ")}.` },
  ];
  const W = 680, bx = 40, bw = 440, gap = 16;
  let yy = 6;
  const pos = blocks.map((b) => { const y0 = yy; yy += (b.h ?? 40) + gap; return y0; });
  const svg = svgOf("m-diagram", W, yy);
  const show = (b: Block) => {
    $("m-detail")!.innerHTML = `<b style="color:var(${ROLE_VAR[b.role]})">${b.title}</b> <span class="shape">${b.shape}</span>${b.params ? ` <span class="shape">${fmt(b.params)} weights</span>` : ""}<br>${b.text}`;
    svg.selectAll(".hit").classed("on", false);
    svg.select(`#blk-${b.id}`).classed("on", true);
  };
  blocks.forEach((b, i) => {
    const y0 = pos[i], h = b.h ?? 40, v = ROLE_VAR[b.role];
    if (i > 0) svg.append("path").attr("d", `M${bx + bw / 2},${y0 - gap + 2} V${y0 - 3}`).attr("stroke", cssVar("--axis")).attr("stroke-width", 1.5).attr("marker-end", "");
    const g = svg.append("g").attr("class", "hit").attr("id", `blk-${b.id}`);
    g.append("rect").attr("x", bx).attr("y", y0).attr("width", bw).attr("height", h).attr("rx", 8).style("fill", mix(v, b.role === "flow" ? 10 : 16)).style("stroke", cssVar(v));
    g.append("rect").attr("x", bx).attr("y", y0).attr("width", 6).attr("height", h).attr("rx", 3).style("fill", cssVar(v));
    g.append("text").attr("x", bx + 16).attr("y", y0 + 25).style("fill", cssVar("--ink")).style("font-size", "13.5px").style("font-weight", "600").text(b.title);
    if (b.id === "enc") {
      const sub: [string, number][] = [["LN → attention → +", P_ATT / c.n_layers], ["LN → feed-forward → +", P_FFN / c.n_layers]];
      sub.forEach(([t, n], k) => {
        g.append("rect").attr("x", bx + 16 + k * 210).attr("y", y0 + 40).attr("width", 200).attr("height", 38).attr("rx", 6).style("fill", mix("--knowledge", 28 + k * 14));
        g.append("text").attr("x", bx + 26 + k * 210).attr("y", y0 + 56).style("fill", cssVar("--ink")).style("font-size", "12.5px").text(t);
        g.append("text").attr("x", bx + 26 + k * 210).attr("y", y0 + 71).style("font-size", "11.5px").text(`${fmt(n)} weights per layer`);
      });
    }
    g.append("text").attr("x", bx + bw + 12).attr("y", y0 + 18).style("font", `12px ${cssVar("--mono")}`).style("fill", cssVar("--ink-2")).text(b.shape);
    if (b.params) g.append("text").attr("x", bx + bw + 12).attr("y", y0 + 34).style("font-size", "11.5px").style("fill", cssVar(v)).text(`${fmt(b.params)} w · ${pct(b.params)}`);
    g.on("mouseenter", () => show(b)).on("click", () => show(b));
  });
  // left rail: which blocks set the range
  svg.append("text").attr("x", 14).attr("y", pos[2] + 20).attr("transform", `rotate(-90 14 ${pos[2] + 20})`).attr("text-anchor", "middle").style("font-size", "11px").style("fill", cssVar("--range")).text("input range");
  svg.append("text").attr("x", 14).attr("y", pos[7] + 20).attr("transform", `rotate(-90 14 ${pos[7] + 20})`).attr("text-anchor", "middle").style("font-size", "11px").style("fill", cssVar("--range")).text("output range");

  // ---- 2. range table
  const rows: [string, string, string, string][] = [
    ["How far back it reads", `<span class="c-range">position table</span> rows × <span class="c-range">patch length</span>`, `${NP} × ${P} = ${T} steps`, `Retrain. Our 2048-step pilot used patches of 64, keeping ${NP} tokens and the same cost.`],
    ["How far ahead it predicts", `<span class="c-range">head</span> output size`, `${H} steps`, "Retrain the head, or feed the forecast back in 64-step blocks (section 6)."],
    ["Which quantiles", `<span class="c-range">head</span> output size`, `${K} levels, ${c.quantile_levels[0]} to ${c.quantile_levels[K - 1]}`, "Retrain the head; interpolate between levels for others."],
    ["Amplitude and unit", `<span class="c-fixed">scaler</span>`, "any", "Nothing to do."],
    ["Missing values", `<span class="c-fixed">mask channel</span>`, "any pattern, ≥ 1 observed step", "Nothing to do."],
    ["Sampling frequency", "nothing: timestamps are not an input", "inferred from the shape", "Add calendar features (not tested)."],
  ];
  $("m-range")!.innerHTML = `<table><thead><tr><th>Limit</th><th>Set by</th><th>Value</th><th>To change it</th></tr></thead><tbody>${rows.map((r) => `<tr>${r.map((x) => `<td>${x}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  $("m-inv")!.textContent = `${(tr.scale_invariance_rel_err * 100).toPrecision(2)}%`;

  // ---- 3. parameters
  {
    const parts: [string, number, string][] = [["feed-forward", P_FFN, "--knowledge"], ["attention", P_ATT, "--knowledge"], ["head", P_HEAD, "--range"], ["patch embedding", P_EMB, "--knowledge"], ["position table", P_POS, "--range"], ["norms + query", M.n_params - P_FFN - P_ATT - P_HEAD - P_EMB - P_POS, "--fixed"]];
    const W2 = 680, Hh = 96, svg2 = svgOf("m-params", W2, Hh), x = scaleLinear().domain([0, M.n_params]).range([0, W2]);
    let acc = 0;
    parts.forEach(([lab, n, v], i) => {
      const g = svg2.append("g");
      g.append("rect").attr("x", x(acc) + 0.5).attr("y", 10).attr("width", Math.max(1, x(acc + n) - x(acc) - 1)).attr("height", 40).attr("rx", 3).style("fill", mix(v, i === 1 ? 50 : i === 3 ? 35 : 75));
      if (x(n) > 60) {
        g.append("text").attr("x", x(acc) + 6).attr("y", 68).attr("class", "lbl-strong").style("font-size", "12px").text(lab);
        g.append("text").attr("x", x(acc) + 6).attr("y", 84).style("font-size", "11.5px").text(`${fmt(n)} · ${pct(n)}`);
      }
      tip(g.node()!, `<b>${lab}</b>: ${fmt(n)} weights (${pct(n)})`);
      acc += n;
    });
    $("m-params-cap")!.innerHTML = `${fmt(M.n_params)} weights in total. <span class="c-knowledge">Green</span>: learned knowledge. <span class="c-range">Violet</span>: weights whose shape is set by the range (head, position table). <span class="c-fixed">Grey</span>: layer norms and the query token, ${fmt(parts[5][1])} weights.`;
  }

  // ---- 4. the stepper
  const iMed = tr.i_med, ctx = tr.context, z = tr.z;
  const STEPS: { title: string; shape: string; text: string; draw: () => void }[] = [];
  const W3 = 680;
  const ctxChart = (vals: N[], id: string, Hh: number, opts: { loc?: number; scale?: number; label: (v: number) => string; colour: string }) => {
    const m = { l: 56, r: 10, t: 14, b: 26 }, s = svgOf(id, W3, Hh);
    const x = scaleLinear().domain([0, T]).range([m.l, W3 - m.r]);
    const [lo, hi] = extent(vals);
    const y = scaleLinear().domain([lo, hi]).nice().range([Hh - m.b, m.t]);
    yAxis(s, y, m.l, W3 - m.r, 4, opts.label);
    if (opts.loc != null && opts.scale != null) {
      s.append("rect").attr("x", m.l).attr("width", W3 - m.r - m.l).attr("y", y(opts.loc + opts.scale)).attr("height", y(opts.loc - opts.scale) - y(opts.loc + opts.scale)).style("fill", mix("--fixed", 16));
      s.append("line").attr("x1", m.l).attr("x2", W3 - m.r).attr("y1", y(opts.loc)).attr("y2", y(opts.loc)).style("stroke", cssVar("--fixed")).attr("stroke-width", 1.5).attr("stroke-dasharray", "5 3");
      s.append("text").attr("x", W3 - m.r - 4).attr("y", y(opts.loc) - 6).attr("text-anchor", "end").style("fill", cssVar("--fixed")).style("font-weight", "600").text(`loc = ${fmt(opts.loc)} · band = ± scale`);
    }
    s.append("path").attr("d", pathOf((i) => x(i), y, vals)).attr("fill", "none").attr("stroke", opts.colour).attr("stroke-width", 1.4);
    for (let d0 = 0; d0 <= T; d0 += 128) s.append("text").attr("x", x(d0)).attr("y", Hh - 8).attr("text-anchor", "middle").text(d0 === T ? "now" : `−${T - d0} h`);
    return { s, x, y };
  };
  const heat = (id: string, M2: number[][], opts: { cell: [number, number]; colLabel?: (j: number) => string; rowLabel?: (i: number) => string; ml?: number; mt?: number; max?: number; highlightCol?: number }) => {
    const [cw, ch] = opts.cell, ml = opts.ml ?? 40, mt = opts.mt ?? 6;
    const nr = M2.length, nc = M2[0].length, Wd = ml + nc * cw + 6, Hd = mt + nr * ch + 24;
    const s = svgOf(id, Math.max(Wd, W3), Hd);
    const mx = opts.max ?? Math.max(...M2.flat().map(Math.abs));
    M2.forEach((row, i) => row.forEach((v, j) => {
      s.append("rect").attr("x", ml + j * cw).attr("y", mt + i * ch).attr("width", cw - 0.6).attr("height", ch - 0.6).style("fill", diverging(v, mx));
    }));
    if (opts.highlightCol != null) s.append("rect").attr("x", ml + opts.highlightCol * cw - 1).attr("y", mt - 1).attr("width", cw + 1).attr("height", nr * ch + 1).attr("fill", "none").style("stroke", cssVar("--knowledge")).attr("stroke-width", 2);
    return { s, ml, mt, cw, ch, nr, nc };
  };
  const tools = $("m-step-tools")!;

  STEPS.push({
    title: "1. The raw input", shape: `(1, ${T}) MW`,
    text: `512 hours of national demand, in megawatts, ending just before ${tr.target_start.slice(0, 16).replace("T", " ")} UTC, where the forecast starts. The <span class="c-fixed">scaler</span> computes two numbers from it: <b>loc = ${fmt(tr.loc)} MW</b> (the median) and <b>scale = ${fmt(tr.scale)} MW</b> (mean distance to the median). They are the only things the model keeps about the unit and level of this series.`,
    draw: () => { ctxChart(ctx, "m-step-svg", 240, { loc: tr.loc, scale: tr.scale, label: (v) => `${(v / 1000).toFixed(0)} GW`, colour: cssVar("--ink-2") }); },
  });
  STEPS.push({
    title: "2. Scaled", shape: `(1, ${T})`,
    text: `z = asinh((x − loc) / scale). The median maps to 0, one scale to about ±0.88. asinh is close to linear near 0 and compresses large excursions, so a spike cannot dominate. The network will never see megawatts.`,
    draw: () => { ctxChart(z, "m-step-svg", 240, { loc: 0, scale: Math.asinh(1), label: (v) => v.toFixed(1), colour: cssVar("--flow") }); },
  });
  STEPS.push({
    title: `3. Cut into ${NP} patches`, shape: `(1, ${NP}, ${P}) + mask`,
    text: `A reshape: column j is hours 16j to 16j + 15. Read the heatmap left to right like the chart above: <span style="color:var(--pos);font-weight:600">orange</span> above the median, <span style="color:var(--flow);font-weight:600">blue</span> below. The daily cycle shows as a repeating pattern every 1.5 columns. The mask (${P} more rows, not drawn) is all ones: no hour is missing in this window.`,
    draw: () => {
      const cols = tr.patch_inputs.map((r) => r.slice(0, P)), M2 = cols[0].map((_, i) => cols.map((r) => r[i]));
      const h = heat("m-step-svg", M2, { cell: [19, 13], ml: 56, mt: 8 });
      h.s.append("text").attr("x", h.ml - 6).attr("y", h.mt + 10).attr("text-anchor", "end").text("hour 0");
      h.s.append("text").attr("x", h.ml - 6).attr("y", h.mt + h.nr * h.ch).attr("text-anchor", "end").text("hour 15");
      for (const j of [0, 8, 16, 24, 31]) h.s.append("text").attr("x", h.ml + j * h.cw + h.cw / 2).attr("y", h.mt + h.nr * h.ch + 16).attr("text-anchor", "middle").text(`p${j + 1}`);
    },
  });
  STEPS.push({
    title: "4. Embedded into tokens", shape: `(1, ${NP + 1}, ${d})`,
    text: `The <span class="c-knowledge">patch embedding</span> turns each patch's 32 inputs into ${d} numbers (rows). The <span class="c-range">position table</span> is added so attention knows where each token sits. The last column, outlined, is the <span class="c-knowledge">query token</span>: the same ${d} learned numbers for every series. It carries no information about this series yet.`,
    draw: () => {
      const cols = [...tr.embeddings, tr.query], M2 = cols[0].map((_, i) => cols.map((r) => r[i]));
      const mx = Math.max(...tr.embeddings.flat().map(Math.abs)) * 0.6;
      const h = heat("m-step-svg", M2, { cell: [19, 2.4], ml: 56, mt: 8, max: mx, highlightCol: NP });
      h.s.append("text").attr("x", h.ml - 6).attr("y", h.mt + 10).attr("text-anchor", "end").text("dim 1");
      h.s.append("text").attr("x", h.ml - 6).attr("y", h.mt + h.nr * h.ch).attr("text-anchor", "end").text(`dim ${d}`);
      for (const j of [0, 8, 16, 24]) h.s.append("text").attr("x", h.ml + j * h.cw + h.cw / 2).attr("y", h.mt + h.nr * h.ch + 16).attr("text-anchor", "middle").text(`p${j + 1}`);
      h.s.append("text").attr("x", h.ml + NP * h.cw + h.cw / 2).attr("y", h.mt + h.nr * h.ch + 16).attr("text-anchor", "middle").style("fill", cssVar("--knowledge")).style("font-weight", "600").text("query");
    },
  });
  let layer = c.n_layers - 1;
  STEPS.push({
    title: "5. Attention: where the query token looks", shape: `(1, ${c.n_heads} heads, ${NP + 1}, ${NP + 1})`,
    text: `In every layer, each of the ${c.n_heads} heads gives the query token a set of weights over all ${NP + 1} tokens (summing to 1) and copies in a weighted mix of them. Bars show those weights under the patches they point to. Switch layers: this is the routing the <span class="c-knowledge">attention weights</span> learned, applied to this one series.`,
    draw: () => {
      tools.innerHTML = `<span class="small muted">layer</span>` + Array.from({ length: c.n_layers }, (_, i) => `<button type="button" data-l="${i}" aria-pressed="${i === layer}">${i + 1}</button>`).join("");
      tools.querySelectorAll<HTMLButtonElement>("button").forEach((b) => (b.onclick = () => { layer = +b.dataset.l!; STEPS[4].draw(); }));
      const A = tr.layers[layer].query_attention, rowH = 46, top = 110, m = { l: 56, r: 10 };
      const Hh = top + c.n_heads * rowH + 24, s = svgOf("m-step-svg", W3, Hh);
      const x = scaleLinear().domain([0, NP + 1]).range([m.l, W3 - m.r]);
      const [lo, hi] = extent(z), y = scaleLinear().domain([lo, hi]).range([top - 16, 8]);
      s.append("path").attr("d", pathOf((i) => x((i / T) * NP), y, z)).attr("fill", "none").attr("stroke", cssVar("--ink-2")).attr("stroke-width", 1);
      s.append("text").attr("x", m.l - 6).attr("y", 20).attr("text-anchor", "end").text("input");
      const mx = Math.max(...A.flat());
      A.forEach((row, hd) => {
        const y0 = top + hd * rowH;
        s.append("text").attr("x", m.l - 6).attr("y", y0 + rowH / 2 + 4).attr("text-anchor", "end").text(`head ${hd + 1}`);
        row.forEach((w, j) => {
          const bh = (rowH - 8) * (w / mx), g = s.append("g");
          g.append("rect").attr("x", x(j) + 1).attr("y", y0 + rowH - 4 - bh).attr("width", x(1) - x(0) - 2).attr("height", Math.max(0.5, bh)).style("fill", j === NP ? cssVar("--knowledge") : cssVar("--flow"));
          tip(g.node()!, `head ${hd + 1} → ${j === NP ? "the query token itself" : `patch ${j + 1} (hours −${T - 16 * j} to −${T - 16 * j - 15})`}: <b>${(100 * w).toFixed(1)}%</b>`);
        });
      });
      s.append("text").attr("x", x(NP + 0.5)).attr("y", Hh - 6).attr("text-anchor", "middle").style("fill", cssVar("--knowledge")).text("query");
      s.append("text").attr("x", x(NP / 2)).attr("y", Hh - 6).attr("text-anchor", "middle").text("← older patches · recent patches →");
    },
  });
  STEPS.push({
    title: "6. What each layer adds to the query token", shape: `(1, ${d}) per layer`,
    text: `A transformer layer does not overwrite its input: it adds to it (the residual stream). Bars show the size of what <span class="c-knowledge">attention</span> (reading the patches) and the <span class="c-knowledge">feed-forward block</span> (transforming the token with stored patterns) add to the query token in each layer. Information about this series flows into the query token through attention only.`,
    draw: () => {
      const Hh = 220, m = { l: 56, r: 10, t: 14, b: 40 }, s = svgOf("m-step-svg", W3, Hh);
      const vals = tr.layers.flatMap((L) => [L.attn_update[NP], L.ffn_update[NP]]);
      const y = scaleLinear().domain([0, Math.max(...vals)]).nice().range([Hh - m.b, m.t]);
      yAxis(s, y, m.l, W3 - m.r, 4, (v) => v.toFixed(0));
      const bw = (W3 - m.l - m.r) / (c.n_layers * 2 + c.n_layers);
      tr.layers.forEach((L, i) => {
        [L.attn_update[NP], L.ffn_update[NP]].forEach((v, k) => {
          const xx = m.l + (i * 3 + k + 0.5) * bw;
          s.append("rect").attr("x", xx).attr("y", y(v)).attr("width", bw - 4).attr("height", y(0) - y(v)).attr("rx", 2).style("fill", mix("--knowledge", k ? 85 : 50));
          s.append("text").attr("x", xx + bw / 2 - 2).attr("y", y(v) - 4).attr("text-anchor", "middle").style("font-size", "11px").text(v.toFixed(1));
        });
        s.append("text").attr("x", m.l + (i * 3 + 1.5) * bw).attr("y", Hh - m.b + 16).attr("text-anchor", "middle").attr("class", "lbl").text(`layer ${i + 1}`);
      });
      s.append("text").attr("x", m.l).attr("y", Hh - 6).style("font-size", "11.5px").text("light: attention update · dark: feed-forward update (vector length)");
    },
  });
  STEPS.push({
    title: "7. The head: 128 numbers → 13 × 64", shape: `(1, ${d}) → (1, ${K * H}) → (1, ${K}, ${H})`,
    text: `The <span class="c-range">head</span> multiplies the final query token by a ${d} × ${K * H} matrix. Top: the ${K} × ${H} raw outputs (rows = levels). The median row (outlined) is used as is. Every other row is passed through softplus, which makes it positive, and summed outward from the median: the gaps between quantiles. Bottom: the result in scaled space. The quantiles cannot cross.`,
    draw: () => {
      const mx = Math.max(...tr.head_raw.flat().map(Math.abs));
      const cw = 9.4, ch = 7, ml = 56, mt = 6, Hh = mt + K * ch + 200, s = svgOf("m-step-svg", W3, Hh);
      tr.head_raw.forEach((row, i) => row.forEach((v, j) => s.append("rect").attr("x", ml + j * cw).attr("y", mt + i * ch).attr("width", cw - 0.5).attr("height", ch - 0.5).style("fill", diverging(v, mx))));
      s.append("rect").attr("x", ml - 1).attr("y", mt + iMed * ch - 1).attr("width", H * cw + 1).attr("height", ch + 1).attr("fill", "none").style("stroke", cssVar("--ink")).attr("stroke-width", 1.5);
      s.append("text").attr("x", ml - 6).attr("y", mt + 8).attr("text-anchor", "end").text("q0.01");
      s.append("text").attr("x", ml - 6).attr("y", mt + K * ch).attr("text-anchor", "end").text("q0.99");
      const top = mt + K * ch + 22, x = scaleLinear().domain([0, H - 1]).range([ml, ml + H * cw]);
      const [lo, hi] = extent(...tr.z_quantiles), y = scaleLinear().domain([lo, hi]).nice().range([Hh - 22, top]);
      yAxis(s, y, ml, ml + H * cw, 3, (v) => v.toFixed(1));
      tr.z_quantiles.forEach((q, k) => s.append("path").attr("d", line<number>().x((_, i) => x(i)).y((v) => y(v))(q) ?? "").attr("fill", "none").style("stroke", cssVar("--flow")).attr("stroke-width", k === iMed ? 2.2 : 0.9).attr("opacity", k === iMed ? 1 : 0.6));
      s.append("text").attr("x", ml + H * cw + 8).attr("y", y(tr.z_quantiles[iMed][H - 1]) + 4).style("font-size", "11.5px").text("median");
      s.append("text").attr("x", ml + H * cw + 8).attr("y", y(tr.z_quantiles[K - 1][H - 1]) + 4).style("font-size", "11.5px").text("0.99");
      s.append("text").attr("x", ml + H * cw + 8).attr("y", y(tr.z_quantiles[0][H - 1]) + 4).style("font-size", "11.5px").text("0.01");
    },
  });
  STEPS.push({
    title: "8. Back to megawatts", shape: `(1, ${K}, ${H}) MW`,
    text: `The <span class="c-fixed">unscale</span> step applies sinh, then × ${fmt(tr.scale)} + ${fmt(tr.loc)}, with the scaler's numbers from step 1. Shown: the last week of context, the 10–90% and 30–70% ranges and the median, and what actually happened (black). This forecast took one pass through the network.`,
    draw: () => {
      const back = 168, Hh = 260, m = { l: 56, r: 10, t: 14, b: 26 }, s = svgOf("m-step-svg", W3, Hh);
      const x = scaleLinear().domain([-back, H]).range([m.l, W3 - m.r]);
      const hist = ctx.slice(T - back);
      const [lo, hi] = extent(hist, tr.future, tr.quantiles[0], tr.quantiles[K - 1]);
      const y = scaleLinear().domain([lo, hi]).nice().range([Hh - m.b, m.t]);
      yAxis(s, y, m.l, W3 - m.r, 4, (v) => `${(v / 1000).toFixed(0)} GW`);
      s.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", m.t).attr("y2", Hh - m.b).attr("stroke", cssVar("--axis")).attr("stroke-dasharray", "4 3");
      s.append("path").attr("d", pathOf((i) => x(i - back), y, hist)).attr("fill", "none").attr("stroke", cssVar("--ink-2")).attr("stroke-width", 1.2);
      fan(s, tr.quantiles, (i) => x(i), y, cssVar("--student"), iMed);
      s.append("path").attr("d", pathOf((i) => x(i), y, tr.future)).attr("fill", "none").attr("stroke", cssVar("--ink")).attr("stroke-width", 1.4);
      for (const t of [-168, -96, -24, 0, 24, 48]) s.append("text").attr("x", x(t)).attr("y", Hh - 8).attr("text-anchor", "middle").text(t === 0 ? "now" : `${t > 0 ? "+" : ""}${t} h`);
    },
  });

  let step = 0;
  const go = (i: number) => {
    step = Math.max(0, Math.min(STEPS.length - 1, i));
    const st = STEPS[step];
    $("m-stepper")!.innerHTML = STEPS.map((_, k) => `<button type="button" data-k="${k}" aria-pressed="${k === step}">${k + 1}</button>`).join("");
    $("m-stepper")!.querySelectorAll<HTMLButtonElement>("button").forEach((b) => (b.onclick = () => go(+b.dataset.k!)));
    $("m-step-title")!.textContent = st.title;
    $("m-step-shape")!.textContent = st.shape;
    $("m-step-text")!.innerHTML = st.text;
    tools.innerHTML = "";
    st.draw();
    ($("m-prev") as HTMLButtonElement).disabled = step === 0;
    ($("m-next") as HTMLButtonElement).disabled = step === STEPS.length - 1;
  };
  $("m-prev")!.onclick = () => go(step - 1);
  $("m-next")!.onclick = () => go(step + 1);
  go(0);

  // ---- 6. improvement axes (paired ratios from the study tables; < 1 = lower error)
  const axes: [string, string, string, string][] = [
    ["More training windows", "30k → 300k: −13%; 300k → 1M: −11% (fresh electricity)", "measured, largest lever", "stage_c_lite, stage_c2"],
    ["Bigger network", "0.22M → 0.95M: −7% fresh electricity [0.903, 0.959], −8% shapes, −3% public data", "measured", "stage_c_lite"],
    ["Bigger still (2.9M)", "TBD: run in progress", "running", "stage_c7_size"],
    ["Longer context (2048, patch 64)", "−6% on public data [0.912, 0.967]; no change on electricity, 0.992 [0.964, 1.020] (300k windows)", "helps on long-cycle data", "stage_c4_ctx"],
    ["Longer horizon (64-step blocks fed back)", "On electricity the gap to the teacher stays flat up to 512 steps; on shapes it grows from 1.02 to 1.47 (an earlier student, same architecture)", "a multi-step head would help on shapes", "long_horizon"],
    ["Covariates (temperature), several series at once, calendar features", "not tested", "open", "–"],
  ];
  $("m-axes")!.innerHTML = `<table><thead><tr><th>Lever</th><th>What we measured</th><th>Verdict</th><th>Source (results/)</th></tr></thead><tbody>${axes.map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td class="src">${r[3]}</td></tr>`).join("")}</tbody></table>`;
}
