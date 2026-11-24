"use client";

// Small multi-series line chart over a rolling time window (hand-rolled SVG, like RollingProbabilityChart).
// null values break the line instead of being drawn as zero.

import { isFiniteNumber } from "@/lib/format";
import { useElementWidth } from "@/lib/useElementWidth";

export interface Series {
  label: string;
  color: string;
  values: (number | null)[];
}

const HEIGHT = 170;
const PAD_L = 40;
const PAD_R = 10;
const PAD_T = 10;
const PAD_B = 22;

export default function TimeSeriesChart({
  times,
  series,
  spanS = 120,
  yMax,
  unit = "",
  label,
}: {
  times: number[]; // unix seconds, ascending, same length as each series
  series: Series[];
  spanS?: number;
  yMax?: number;
  unit?: string;
  label: string;
}) {
  const [ref, width] = useElementWidth<HTMLDivElement>();
  const all = series.flatMap((s) => s.values.filter(isFiniteNumber));

  if (times.length < 2 || all.length === 0) {
    return (
      <div className="empty-state" style={{ height: HEIGHT, display: "grid", placeItems: "center" }}>
        {times.length < 2 ? "Collecting samples…" : "No data reported for this chart"}
      </div>
    );
  }

  const tEnd = times[times.length - 1];
  const tStart = tEnd - spanS;
  const top = yMax ?? Math.max(1, ...all) * 1.15;
  const innerW = Math.max(10, width - PAD_L - PAD_R);
  const innerH = HEIGHT - PAD_T - PAD_B;
  const x = (t: number) => PAD_L + (innerW * (t - tStart)) / spanS;
  const y = (v: number) => PAD_T + innerH * (1 - Math.min(top, Math.max(0, v)) / top);

  const pathFor = (vals: (number | null)[]) => {
    let d = "";
    let pen = false;
    vals.forEach((v, i) => {
      if (!isFiniteNumber(v) || times[i] < tStart) {
        pen = false;
        return;
      }
      d += `${pen ? "L" : "M"}${x(times[i]).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    });
    return d;
  };

  const fmt = (v: number) => (v >= 100 ? Math.round(v).toString() : Number(v.toPrecision(2)).toString());
  const latest = series
    .map((s) => {
      const v = s.values[s.values.length - 1];
      return `${s.label} ${isFiniteNumber(v) ? fmt(v) + unit : "not reported"}`;
    })
    .join(", ");

  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={HEIGHT} viewBox={`0 0 ${width} ${HEIGHT}`} role="img" aria-label={`${label}, last ${spanS} s. Latest: ${latest}.`}>
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <line x1={PAD_L} x2={width - PAD_R} y1={y(top * f)} y2={y(top * f)} stroke="var(--border)" />
            <text className="chart-axis" x={PAD_L - 6} y={y(top * f) + 3} textAnchor="end">
              {fmt(top * f)}
            </text>
          </g>
        ))}
        {series.map((s) => (
          <path key={s.label} d={pathFor(s.values)} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" />
        ))}
        <text className="chart-axis" x={PAD_L} y={HEIGHT - 6}>
          −{spanS}s
        </text>
        <text className="chart-axis" x={width - PAD_R} y={HEIGHT - 6} textAnchor="end">
          now
        </text>
      </svg>
      <div className="chart-legend">
        {series.map((s) => (
          <span key={s.label}>
            <i className="swatch-line" style={{ background: s.color }} /> {s.label}
            {unit && ` (${unit.trim()})`}
          </span>
        ))}
      </div>
    </div>
  );
}
