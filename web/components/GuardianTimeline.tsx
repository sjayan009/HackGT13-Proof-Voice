"use client";

// Per-hop evidence timeline for one call. Upper pane: sequential evidence (0–1.5, alert line at 1.0).
// Lower pane: per-window synthetic probability (filled = usable, hollow = not usable) and skipped hops.
// Hand-rolled SVG, same conventions as RollingProbabilityChart.

import { clamp01, isFiniteNumber, pct, seconds, timeTicks } from "@/lib/format";
import { useElementWidth } from "@/lib/useElementWidth";

export interface TimelinePoint {
  t_ms: number;
  evidence: number | null;
  p: number | null; // window synthetic_probability; null for skipped hops / not stored
  usable: boolean | null;
  skipped: boolean;
  action?: string;
}

const PAD_L = 38;
const PAD_R = 12;
const TOP_H = 120;
const GAP = 34;
const BOT_H = 110;
const SKIP_H = 14;
const AXIS_H = 22;
const HEIGHT = TOP_H + GAP + BOT_H + SKIP_H + AXIS_H + 8;
const MIN_SPAN_MS = 10_000;
const EV_MAX = 1.5;

export default function GuardianTimeline({ points }: { points: TimelinePoint[] }) {
  const [ref, width] = useElementWidth<HTMLDivElement>();

  if (points.length === 0) {
    return <div className="empty-state">No analysis has arrived for this call yet.</div>;
  }

  const maxT = Math.max(MIN_SPAN_MS, ...points.map((p) => p.t_ms));
  const innerW = Math.max(10, width - PAD_L - PAD_R);
  const x = (t: number) => PAD_L + (innerW * t) / maxT;
  const yEv = (v: number) => 4 + TOP_H * (1 - Math.max(0, Math.min(EV_MAX, v)) / EV_MAX);
  const botTop = 4 + TOP_H + GAP;
  const yP = (v: number) => botTop + BOT_H * (1 - clamp01(v));
  const skipY = botTop + BOT_H + 4;

  const ev = points.filter((p) => isFiniteNumber(p.evidence));
  const evPath = ev.length ? "M" + ev.map((p) => `${x(p.t_ms).toFixed(1)},${yEv(p.evidence!).toFixed(1)}`).join("L") : "";
  const windows = points.filter((p) => !p.skipped && isFiniteNumber(p.p));
  const skipped = points.filter((p) => p.skipped);
  const usableN = windows.filter((w) => w.usable).length;
  const ticks = timeTicks(maxT, Math.max(2, Math.min(8, Math.floor(innerW / 90))));
  const lastEv = ev[ev.length - 1]?.evidence;

  return (
    <div className="chart" ref={ref}>
      <svg
        width={width}
        height={HEIGHT}
        viewBox={`0 0 ${width} ${HEIGHT}`}
        role="img"
        aria-label={`Evidence timeline: ${points.length} half-second steps, ${windows.length} windows scored (${usableN} usable), ${skipped.length} skipped. Latest evidence ${
          isFiniteNumber(lastEv) ? lastEv.toFixed(2) : "not available"
        }; the alert line is at 1.0.`}
      >
        {/* Evidence pane */}
        {[0, 0.5, 1, 1.5].map((v) => (
          <g key={`e${v}`}>
            <line x1={PAD_L} x2={width - PAD_R} y1={yEv(v)} y2={yEv(v)} stroke="var(--border)" />
            <text className="chart-axis" x={PAD_L - 8} y={yEv(v) + 3} textAnchor="end">
              {v}
            </text>
          </g>
        ))}
        <line
          x1={PAD_L}
          x2={width - PAD_R}
          y1={yEv(1)}
          y2={yEv(1)}
          stroke="var(--red)"
          strokeWidth={1.5}
          strokeDasharray="5 3"
        />
        <text className="chart-axis" x={width - PAD_R} y={yEv(1) - 4} textAnchor="end" fill="var(--red)">
          alert line
        </text>
        {evPath && <path d={evPath} fill="none" stroke="var(--accent)" strokeWidth={2.5} strokeLinejoin="round" />}

        <text className="chart-axis" x={PAD_L} y={botTop - 8}>
          Per-window synthetic probability
        </text>
        {/* Per-window probability pane */}
        {[0, 0.5, 1].map((v) => (
          <g key={`p${v}`}>
            <line x1={PAD_L} x2={width - PAD_R} y1={yP(v)} y2={yP(v)} stroke="var(--border)" />
            <text className="chart-axis" x={PAD_L - 8} y={yP(v) + 3} textAnchor="end">
              {v === 0.5 ? ".5" : v}
            </text>
          </g>
        ))}
        {windows.map((w, i) =>
          w.usable ? (
            <circle key={i} cx={x(w.t_ms)} cy={yP(w.p!)} r={3} fill="var(--accent)" />
          ) : (
            <circle key={i} cx={x(w.t_ms)} cy={yP(w.p!)} r={3} fill="none" stroke="var(--text-faint)" strokeWidth={1.25} />
          )
        )}
        {windows.length === 0 && (
          <text className="chart-axis" x={PAD_L + 8} y={yP(0.5)}>
            No per-window scores available
          </text>
        )}
        {skipped.map((s, i) => (
          <rect key={i} x={x(s.t_ms) - 1} y={skipY} width={2} height={SKIP_H - 4} fill="var(--text-faint)" opacity={0.7}>
            <title>{`${seconds(s.t_ms)}: skipped (${s.action ?? "no window"})`}</title>
          </rect>
        ))}

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
          <i className="swatch-line" style={{ background: "var(--accent)" }} /> Evidence (0–1.5)
        </span>
        <span>
          <i className="swatch-dash" style={{ borderColor: "var(--red)" }} /> Alert line (1.0)
        </span>
        <span>
          <i className="swatch" style={{ background: "var(--accent)", borderRadius: 999 }} /> Usable window score
        </span>
        <span>
          <i className="swatch" style={{ border: "1px solid var(--text-faint)", borderRadius: 999 }} /> Not usable (noise,
          silence, packet loss)
        </span>
        <span>
          <i className="swatch" style={{ background: "var(--text-faint)", width: 3 }} /> Skipped step
        </span>
      </div>
      <details className="disclosure">
        <summary>Show as a table</summary>
        <div className="table-scroll" style={{ maxHeight: 280 }}>
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Time</th>
                <th scope="col" className="num">Evidence</th>
                <th scope="col" className="num">Window score</th>
                <th scope="col">Window</th>
              </tr>
            </thead>
            <tbody>
              {points.map((p, i) => (
                <tr key={i}>
                  <td className="mono">{seconds(p.t_ms)}</td>
                  <td className="num mono">{isFiniteNumber(p.evidence) ? p.evidence.toFixed(2) : "—"}</td>
                  <td className="num mono">{pct(p.p)}</td>
                  <td>{p.skipped ? `skipped (${p.action ?? "—"})` : p.usable == null ? "—" : p.usable ? "usable" : "not usable"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}
