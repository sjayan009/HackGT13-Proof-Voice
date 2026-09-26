import type { SuspiciousRegion, TimelineWindow } from "@/lib/types";

const WIDTH = 800;
const HEIGHT = 220;
const PAD_L = 40;
const PAD_R = 12;
const PAD_T = 12;
const PAD_B = 24;

export default function TrustTimelineChart({
  timeline,
  suspiciousRegions,
  decisionThreshold,
}: {
  timeline: TimelineWindow[];
  suspiciousRegions: SuspiciousRegion[];
  decisionThreshold: number | null;
}) {
  if (!timeline || timeline.length === 0) {
    return <div className="empty-state">No per-window timeline data.</div>;
  }

  const maxT = Math.max(...timeline.map((w) => w.end_ms));
  const minT = 0;
  const innerW = WIDTH - PAD_L - PAD_R;
  const innerH = HEIGHT - PAD_T - PAD_B;

  const x = (ms: number) =>
    PAD_L + (innerW * (ms - minT)) / Math.max(1, maxT - minT);
  const y = (p: number) => PAD_T + innerH * (1 - Math.max(0, Math.min(1, p)));

  const points = timeline.map((w) => {
    const mid = (w.start_ms + w.end_ms) / 2;
    return `${x(mid)},${y(w.synthetic_probability)}`;
  });

  const linePath = "M " + points.join(" L ");
  const areaPath =
    `M ${x((timeline[0].start_ms + timeline[0].end_ms) / 2)},${y(0)} ` +
    `L ${points.join(" L ")} ` +
    `L ${x((timeline[timeline.length - 1].start_ms + timeline[timeline.length - 1].end_ms) / 2)},${y(0)} Z`;

  const ticks = 5;
  const tickVals = Array.from({ length: ticks + 1 }, (_, i) =>
    Math.round((maxT * i) / ticks)
  );

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      width="100%"
      height={HEIGHT}
      role="img"
      aria-label="Synthetic probability over time"
    >
      {/* y gridlines at 0, 0.5, 1 */}
      {[0, 0.5, 1].map((p) => (
        <g key={p}>
          <line
            x1={PAD_L}
            x2={WIDTH - PAD_R}
            y1={y(p)}
            y2={y(p)}
            stroke="var(--border)"
            strokeWidth={1}
          />
          <text x={4} y={y(p) + 4} fontSize="10" fill="var(--text-faint)">
            {p}
          </text>
        </g>
      ))}

      {/* suspicious regions shaded */}
      {suspiciousRegions?.map((r, i) => (
        <rect
          key={i}
          x={x(r.start_ms)}
          y={PAD_T}
          width={Math.max(1, x(r.end_ms) - x(r.start_ms))}
          height={innerH}
          fill="var(--red)"
          opacity={0.12}
        >
          <title>
            {`Suspicious ${r.start_ms}-${r.end_ms}ms — peak ${(
              r.peak_probability * 100
            ).toFixed(0)}% — ${r.reason}`}
          </title>
        </rect>
      ))}

      {/* threshold line */}
      {decisionThreshold != null && (
        <line
          x1={PAD_L}
          x2={WIDTH - PAD_R}
          y1={y(decisionThreshold)}
          y2={y(decisionThreshold)}
          stroke="var(--amber)"
          strokeWidth={1.5}
          strokeDasharray="4 3"
        />
      )}

      {/* area + line */}
      <path d={areaPath} fill="var(--accent)" opacity={0.12} />
      <path d={linePath} fill="none" stroke="var(--accent)" strokeWidth={2} />

      {/* x axis ticks */}
      {tickVals.map((t) => (
        <text
          key={t}
          x={x(t)}
          y={HEIGHT - 6}
          fontSize="10"
          fill="var(--text-faint)"
          textAnchor="middle"
        >
          {(t / 1000).toFixed(1)}s
        </text>
      ))}
    </svg>
  );
}
