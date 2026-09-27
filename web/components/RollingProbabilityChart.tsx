"use client";

import type { AnalysisWindowEvent } from "@/lib/types";
import { clamp01, isFiniteNumber, pct, seconds, timeTicks } from "@/lib/format";
import { useElementWidth } from "@/lib/useElementWidth";

const HEIGHT = 220;
const PAD_L = 34;
const PAD_R = 10;
const PAD_T = 10;
const PAD_B = 24;
// Keep at least this much time on the x-axis so the first seconds don't stretch across the chart.
const MIN_SPAN_MS = 10_000;

export default function RollingProbabilityChart({
  events,
  decisionThreshold,
}: {
  events: AnalysisWindowEvent[];
  decisionThreshold?: number | null;
}) {
  const [ref, width] = useElementWidth<HTMLDivElement>();

  if (!events || events.length === 0) {
    return (
      <div className="empty-state" style={{ height: HEIGHT, display: "grid", placeItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span className="spinner" aria-hidden />
          Waiting for the first 2-second analysis window…
        </div>
      </div>
    );
  }

  const maxT = Math.max(MIN_SPAN_MS, ...events.map((e) => e.t_ms));
  const innerW = Math.max(10, width - PAD_L - PAD_R);
  const innerH = HEIGHT - PAD_T - PAD_B;

  const x = (t: number) => PAD_L + (innerW * t) / maxT;
  const y = (p: number) => PAD_T + innerH * (1 - clamp01(p));
  const path = (vals: [number, number][]) =>
    "M" + vals.map(([a, b]) => `${a.toFixed(1)},${b.toFixed(1)}`).join("L");

  const raw = path(events.map((e) => [x(e.t_ms), y(e.synthetic_probability)]));
  const rolling = path(events.map((e) => [x(e.t_ms), y(e.rolling_probability)]));

  // Uncertainty band: rolling ± (1 − analysis_confidence) · 0.2 — a visual cue, not a statistical interval.
  const spread = (e: AnalysisWindowEvent) => (1 - (e.analysis_confidence ?? 0)) * 0.2;
  const upper = events.map((e) => [x(e.t_ms), y(e.rolling_probability + spread(e))] as [number, number]);
  const lower = events
    .slice()
    .reverse()
    .map((e) => [x(e.t_ms), y(e.rolling_probability - spread(e))] as [number, number]);
  const band = `${path(upper)}L${lower.map(([a, b]) => `${a.toFixed(1)},${b.toFixed(1)}`).join("L")}Z`;

  const last = events[events.length - 1];
  const tickCount = Math.max(2, Math.min(8, Math.floor(innerW / 90)));
  const ticks = timeTicks(maxT, tickCount);

  return (
    <div className="chart" ref={ref}>
      <svg
        width={width}
        height={HEIGHT}
        viewBox={`0 0 ${width} ${HEIGHT}`}
        role="img"
        aria-label={`Rolling synthetic probability, currently ${pct(last.rolling_probability)} after ${seconds(
          last.t_ms
        )}.`}
      >
        {[0, 0.5, 1].map((p) => (
          <g key={p}>
            <line x1={PAD_L} x2={width - PAD_R} y1={y(p)} y2={y(p)} stroke="var(--border)" strokeWidth={1} />
            <text className="chart-axis" x={PAD_L - 8} y={y(p) + 3} textAnchor="end">
              {p === 0.5 ? ".5" : p}
            </text>
          </g>
        ))}

        {isFiniteNumber(decisionThreshold) && (
          <line
            x1={PAD_L}
            x2={width - PAD_R}
            y1={y(decisionThreshold)}
            y2={y(decisionThreshold)}
            stroke="var(--amber)"
            strokeWidth={1.25}
            strokeDasharray="4 3"
          />
        )}

        <path d={band} fill="var(--accent)" opacity={0.12} />
        <path d={raw} fill="none" stroke="var(--text-faint)" strokeWidth={1} opacity={0.7} />
        <path
          d={rolling}
          fill="none"
          stroke="var(--accent)"
          strokeWidth={2.5}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        <circle cx={x(last.t_ms)} cy={y(last.rolling_probability)} r={4.5} fill="var(--accent)" />
        <circle
          cx={x(last.t_ms)}
          cy={y(last.rolling_probability)}
          r={8}
          fill="var(--accent)"
          opacity={0.2}
        />

        {ticks.map((t, i) => (
          <text
            key={i}
            className="chart-axis"
            x={x(t)}
            y={HEIGHT - 6}
            textAnchor={i === 0 ? "start" : x(t) > width - PAD_R - 16 ? "end" : "middle"}
          >
            {seconds(t)}
          </text>
        ))}
      </svg>
      <div className="chart-legend">
        <span>
          <i className="swatch-line" style={{ background: "var(--accent)" }} /> Rolling probability
        </span>
        <span>
          <i className="swatch-line" style={{ background: "var(--text-faint)" }} /> Per-window
        </span>
        <span>
          <i className="swatch" style={{ background: "var(--accent)", opacity: 0.25 }} /> Uncertainty (1 −
          confidence)
        </span>
      </div>
    </div>
  );
}
