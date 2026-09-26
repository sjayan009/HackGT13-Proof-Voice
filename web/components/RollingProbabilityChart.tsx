import type { AnalysisWindowEvent } from "@/lib/types";

const WIDTH = 800;
const HEIGHT = 220;
const PAD_L = 40;
const PAD_R = 12;
const PAD_T = 12;
const PAD_B = 24;

export default function RollingProbabilityChart({
  events,
  decisionThreshold,
}: {
  events: AnalysisWindowEvent[];
  decisionThreshold?: number | null;
}) {
  if (!events || events.length === 0) {
    return <div className="empty-state">Waiting for live data…</div>;
  }

  const maxT = Math.max(...events.map((e) => e.t_ms), 1);
  const innerW = WIDTH - PAD_L - PAD_R;
  const innerH = HEIGHT - PAD_T - PAD_B;

  const x = (ms: number) => PAD_L + (innerW * ms) / maxT;
  const y = (p: number) => PAD_T + innerH * (1 - Math.max(0, Math.min(1, p)));

  const rawPoints = events.map((e) => `${x(e.t_ms)},${y(e.synthetic_probability)}`);
  const rollingPoints = events.map(
    (e) => `${x(e.t_ms)},${y(e.rolling_probability)}`
  );

  // Confidence band: rolling +/- (1 - analysis_confidence) * 0.25 as a soft visual uncertainty cue.
  const bandUpper = events.map((e) => {
    const spread = (1 - (e.analysis_confidence ?? 0)) * 0.2;
    return `${x(e.t_ms)},${y(e.rolling_probability + spread)}`;
  });
  const bandLower = events
    .slice()
    .reverse()
    .map((e) => {
      const spread = (1 - (e.analysis_confidence ?? 0)) * 0.2;
      return `${x(e.t_ms)},${y(e.rolling_probability - spread)}`;
    });
  const bandPath = `M ${bandUpper.join(" L ")} L ${bandLower.join(" L ")} Z`;

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      width="100%"
      height={HEIGHT}
      role="img"
      aria-label="Rolling synthetic probability"
    >
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

      {/* uncertainty band around rolling probability */}
      <path d={bandPath} fill="var(--accent)" opacity={0.1} />

      {/* raw per-window probability, faint */}
      <polyline
        points={rawPoints.join(" ")}
        fill="none"
        stroke="var(--text-faint)"
        strokeWidth={1}
        opacity={0.6}
      />

      {/* rolling probability, prominent */}
      <polyline
        points={rollingPoints.join(" ")}
        fill="none"
        stroke="var(--accent)"
        strokeWidth={2.5}
      />

      {[0, maxT / 2, maxT].map((t, i) => (
        <text
          key={i}
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
