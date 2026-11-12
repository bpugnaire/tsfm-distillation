import { scaleLinear } from "d3-scale";
import { select } from "d3-selection";
import { area, line } from "d3-shape";
import { cssVar } from "./data";

export interface FanOpts {
  context: (number | null)[];
  future?: (number | null)[];
  q?: number[][]; // levels x H
  levels?: number[];
  color: string;
  width: number;
  height: number;
  showContext?: number;
  axes?: boolean;
  bands?: [number, number, number][];
  yDomain?: [number, number]; // fixed y range (draw mode); default: from the data
  layoutH?: number; // reserve this many future steps on the x axis (default: forecast length)
  extraMedian?: { q: number[]; color: string }; // e.g. a precomputed teacher median
}

const DEFAULT_BANDS: [number, number, number][] = [[0.01, 0.99, 0.15], [0.1, 0.9, 0.28], [0.3, 0.7, 0.45]];

export function drawFan(svgEl: SVGSVGElement, o: FanOpts): void {
  const svg = select(svgEl);
  svg.selectAll("*").remove();
  svg.attr("viewBox", `0 0 ${o.width} ${o.height}`).attr("width", "100%");
  const T = o.context.length;
  const show = Math.min(o.showContext ?? T, T);
  const H = o.q ? o.q[0].length : o.future ? o.future.length : 0;
  const layoutH = o.layoutH ?? H;
  const ctx = o.context.slice(T - show);
  const obs = [...ctx, ...(o.future ?? [])].filter((v): v is number => v !== null && Number.isFinite(v));
  let lo = Math.min(...obs), hi = Math.max(...obs);
  if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
  let pad = 0.3 * (hi - lo || Math.abs(hi) || 1);
  if (o.yDomain) { [lo, hi] = o.yDomain; pad = 0; }
  const m = o.axes ? { l: 40, r: 8, t: 8, b: 18 } : { l: 2, r: 2, t: 4, b: 4 };
  const x = scaleLinear().domain([0, show + layoutH]).range([m.l, o.width - m.r]);
  const y = scaleLinear().domain([lo - pad, hi + pad]).range([o.height - m.b, m.t]);
  const clipId = `c${Math.random().toString(36).slice(2)}`;
  svg.append("clipPath").attr("id", clipId).append("rect")
    .attr("x", m.l).attr("y", m.t).attr("width", o.width - m.l - m.r).attr("height", o.height - m.t - m.b);
  const g = svg.append("g").attr("clip-path", `url(#${clipId})`);
  if (o.axes) {
    for (const t of y.ticks(4)) {
      svg.append("line").attr("x1", m.l).attr("x2", o.width - m.r).attr("y1", y(t)).attr("y2", y(t)).attr("stroke", cssVar("--grid"));
      svg.append("text").attr("x", m.l - 6).attr("y", y(t) + 4).attr("text-anchor", "end").text(+t.toPrecision(4));
    }
  }
  g.append("line").attr("x1", x(show - 0.5)).attr("x2", x(show - 0.5)).attr("y1", m.t).attr("y2", o.height - m.b).attr("stroke", cssVar("--axis"));
  const ln = line<number | null>().defined((d) => d !== null && Number.isFinite(d)).x((_, i) => x(i)).y((d) => y(d as number));
  g.append("path").attr("d", ln(ctx)).attr("fill", "none").attr("stroke", cssVar("--ink-2")).attr("stroke-width", 1.2);
  if (o.q && o.levels) {
    const L = o.levels.map((v) => Math.round(v * 1000));
    for (const [a, b, op] of o.bands ?? DEFAULT_BANDS) {
      const ia = L.indexOf(Math.round(a * 1000)), ib = L.indexOf(Math.round(b * 1000));
      if (ia < 0 || ib < 0) continue;
      const ar = area<number>().x((_, i) => x(show + i)).y0((_, i) => y(o.q![ia][i])).y1((_, i) => y(o.q![ib][i]));
      g.append("path").attr("d", ar(o.q[ia])).attr("fill", o.color).attr("opacity", op);
    }
    const im = L.indexOf(500);
    const lm = line<number>().x((_, i) => x(show + i)).y((d) => y(d));
    g.append("path").attr("d", lm(o.q[im])).attr("fill", "none").attr("stroke", o.color).attr("stroke-width", 2);
  }
  if (o.extraMedian) {
    const le = line<number>().x((_, i) => x(show + i)).y((d) => y(d));
    g.append("path").attr("d", le(o.extraMedian.q)).attr("fill", "none").attr("stroke", o.extraMedian.color)
      .attr("stroke-width", 1.6).attr("stroke-dasharray", "6 3");
  }
  if (o.future) {
    const lf = line<number | null>().defined((d) => d !== null).x((_, i) => x(show + i)).y((d) => y(d as number));
    g.append("path").attr("d", lf(o.future.slice(0, H || o.future.length))).attr("fill", "none")
      .attr("stroke", cssVar("--ink")).attr("stroke-width", 1).attr("stroke-dasharray", "3 2");
  }
}
