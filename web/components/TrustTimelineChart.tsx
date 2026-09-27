"use client";

import { useState } from "react";
import type { SuspiciousRegion, TimelineWindow } from "@/lib/types";
import { clamp01, isFiniteNumber, pct, seconds, timeTicks } from "@/lib/format";
import { useElementWidth } from "@/lib/useElementWidth";

const PAD_L = 34;
const PAD_R = 10;
const PAD_T = 8;
const PAD_B = 24;
const PLOT_H = 190;
const WAVE_H = 44;
const WAVE_GAP = 10;

export default function TrustTimelineChart({
  timeline,
  suspiciousRegions,
  decisionThreshold,
  waveformPeaks,
  durationMs,
  playheadMs,
  onSeek,
}: {
  timeline: TimelineWindow[];
  suspiciousRegions: SuspiciousRegion[];
  decisionThreshold: number | null;
  waveformPeaks?: number[] | null;
  durationMs?: number | null;
  playheadMs?: number | null;
  onSeek?: (ms: number) => void;
}) {
  const [ref, width] = useElementWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);

  if (!timeline || timeline.length === 0) {
    return <div className="empty-state">No per-window timeline data.</div>;
  }

  const hasWave = !!waveformPeaks && waveformPeaks.length > 0;
  const top = PAD_T + (hasWave ? WAVE_H + WAVE_GAP : 0);
  const height = top + PLOT_H + PAD_B;
  const maxT = Math.max(
    1,
    ...timeline.map((w) => w.end_ms),
    isFiniteNumber(durationMs) ? durationMs : 0
  );
  const innerW = Math.max(10, width - PAD_L - PAD_R);

  const x = (t: number) => PAD_L + (innerW * t) / maxT;
  const y = (p: number) => top + PLOT_H * (1 - clamp01(p));
  const mid = (w: TimelineWindow) => (w.start_ms + w.end_ms) / 2;

  // Each window covers [start, end], so the curve spans from the first window's start to the last one's end.
  const first = timeline[0];
  const lastW = timeline[timeline.length - 1];
  const pts = [
    [x(first.start_ms), y(first.synthetic_probability)] as const,
    ...timeline.map((w) => [x(mid(w)), y(w.synthetic_probability)] as const),
    [x(lastW.end_ms), y(lastW.synthetic_probability)] as const,
  ];
  const linePath = "M" + pts.map(([a, b]) => `${a.toFixed(1)},${b.toFixed(1)}`).join("L");
  const areaPath = `${linePath}L${pts[pts.length - 1][0].toFixed(1)},${y(0)}L${pts[0][0].toFixed(1)},${y(0)}Z`;

  const tickCount = Math.max(2, Math.min(8, Math.floor(innerW / 90)));
  const ticks = timeTicks(maxT, tickCount);

  const timeAt = (e: React.PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return Math.max(0, Math.min(maxT, ((e.clientX - rect.left - PAD_L) / innerW) * maxT));
  };

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const t = timeAt(e);
    let best = 0;
    let bestD = Infinity;
    timeline.forEach((w, i) => {
      const d = Math.abs(mid(w) - t);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    });
    setHover(best);
  };

  const hw = hover != null ? timeline[hover] : null;
  const peakWindow = timeline.reduce((a, b) => (b.synthetic_probability > a.synthetic_probability ? b : a));
  const gradId = "tl-grad";

  // Waveform bars across the file duration, sharing the timeline x-axis.
  let wave: JSX.Element | null = null;
  if (hasWave) {
    const peaks = waveformPeaks!;
    const maxPeak = Math.max(...peaks, 1e-3);
    const waveSpanMs = isFiniteNumber(durationMs) && durationMs > 0 ? durationMs : maxT;
    const barW = (x(waveSpanMs) - PAD_L) / peaks.length;
    const cy = PAD_T + WAVE_H / 2;
    wave = (
      <g aria-hidden>
        {peaks.map((p, i) => {
          const h = Math.max(1, (p / maxPeak) * (WAVE_H - 2));
          const bx = PAD_L + i * barW;
          const played = isFiniteNumber(playheadMs) && bx <= x(playheadMs);
          return (
            <rect
              key={i}
              x={bx}
              y={cy - h / 2}
              width={Math.max(0.8, barW - 0.6)}
              height={h}
              rx={Math.min(1, barW / 2)}
              fill={played ? "var(--accent)" : "var(--text-faint)"}
              opacity={played ? 0.9 : 0.55}
            />
          );
        })}
      </g>
    );
  }

  return (
    <div className="chart" ref={ref}>
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Synthetic probability over ${seconds(maxT)}: ${timeline.length} windows, peak ${pct(
          peakWindow.synthetic_probability
        )} at ${seconds(mid(peakWindow))}, ${suspiciousRegions?.length ?? 0} suspicious regions.`}
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
        onPointerDown={onSeek ? (e) => onSeek(timeAt(e)) : undefined}
        style={onSeek ? { cursor: "pointer" } : undefined}
      >
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="var(--accent)" stopOpacity={0.28} />
            <stop offset="1" stopColor="var(--accent)" stopOpacity={0} />
          </linearGradient>
        </defs>

        {wave}

        {[0, 0.5, 1].map((p) => (
          <g key={p}>
            <line x1={PAD_L} x2={width - PAD_R} y1={y(p)} y2={y(p)} stroke="var(--border)" strokeWidth={1} />
            <text className="chart-axis" x={PAD_L - 8} y={y(p) + 3} textAnchor="end">
              {p === 0.5 ? ".5" : p}
            </text>
          </g>
        ))}

        {suspiciousRegions?.map((r, i) => (
          <rect
            key={i}
            x={x(r.start_ms)}
            y={PAD_T}
            width={Math.max(2, x(r.end_ms) - x(r.start_ms))}
            height={top - PAD_T + PLOT_H}
            fill="var(--red)"
            opacity={0.12}
            rx={3}
          >
            <title>{`Suspicious ${seconds(r.start_ms)}–${seconds(r.end_ms)} · peak ${pct(
              r.peak_probability
            )} · ${r.reason}`}</title>
          </rect>
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

        <path d={areaPath} fill={`url(#${gradId})`} />
        <path
          d={linePath}
          fill="none"
          stroke="var(--accent)"
          strokeWidth={2}
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        {ticks.map((t, i) => (
          <text
            key={i}
            className="chart-axis"
            x={x(t)}
            y={height - 6}
            textAnchor={i === 0 ? "start" : x(t) > width - PAD_R - 16 ? "end" : "middle"}
          >
            {seconds(t)}
          </text>
        ))}

        {isFiniteNumber(playheadMs) && (
          <g pointerEvents="none" aria-hidden>
            <line
              x1={x(playheadMs)}
              x2={x(playheadMs)}
              y1={PAD_T - 4}
              y2={top + PLOT_H}
              stroke="var(--text)"
              strokeWidth={1.5}
            />
            <circle cx={x(playheadMs)} cy={PAD_T - 4} r={3.5} fill="var(--text)" />
          </g>
        )}

        {hw && (
          <g pointerEvents="none">
            <line
              x1={x(mid(hw))}
              x2={x(mid(hw))}
              y1={PAD_T}
              y2={top + PLOT_H}
              stroke="var(--text-faint)"
              strokeWidth={1}
            />
            <circle
              cx={x(mid(hw))}
              cy={y(hw.synthetic_probability)}
              r={4.5}
              fill="var(--bg-elev)"
              stroke="var(--accent)"
              strokeWidth={2}
            />
          </g>
        )}
      </svg>

      {hw && (
        <div
          className="chart-tooltip"
          style={{
            left: Math.min(Math.max(x(mid(hw)), 80), width - 80),
            top: Math.max(0, y(hw.synthetic_probability) - 64),
          }}
        >
          <div>
            <span className="num">{pct(hw.synthetic_probability, 1)}</span>{" "}
            <span className="muted">p(synthetic)</span>
          </div>
          <div className="muted num">
            {seconds(hw.start_ms)}–{seconds(hw.end_ms)}
            {isFiniteNumber(hw.analysis_confidence) && ` · conf ${pct(hw.analysis_confidence)}`}
          </div>
        </div>
      )}

      <div className="chart-legend">
        <span>
          <i className="swatch-line" style={{ background: "var(--accent)" }} /> Window p(synthetic)
        </span>
        {isFiniteNumber(decisionThreshold) && (
          <span>
            <i className="swatch-dash" /> Decision threshold {decisionThreshold.toFixed(2)}
          </span>
        )}
        {(suspiciousRegions?.length ?? 0) > 0 && (
          <span>
            <i className="swatch" style={{ background: "var(--red)", opacity: 0.35 }} /> Suspicious region
          </span>
        )}
        {hasWave && (
          <span>
            <i className="swatch" style={{ background: "var(--text-faint)", opacity: 0.55 }} /> Waveform
          </span>
        )}
      </div>
    </div>
  );
}
