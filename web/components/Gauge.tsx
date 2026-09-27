"use client";

import { useEffect, useState } from "react";
import type { Status } from "@/lib/types";
import { STATUS_COLOR, clamp01, isFiniteNumber } from "@/lib/format";

export default function Gauge({
  probability,
  status,
  size = 168,
}: {
  probability: number | null;
  status: Status | null;
  size?: number;
}) {
  const valid = isFiniteNumber(probability);
  const target = valid ? clamp01(probability) : 0;
  // Sweep in from zero on mount; later updates animate from the current value.
  const [shown, setShown] = useState(0);
  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(target));
    return () => cancelAnimationFrame(id);
  }, [target]);

  const stroke = 12;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const dash = shown * c;
  const color = status ? STATUS_COLOR[status] : "var(--gray)";
  const pctText = valid ? Math.round(target * 100) : null;

  return (
    <div
      className="gauge-wrap"
      style={{ width: size, height: size }}
      role="img"
      aria-label={
        pctText == null ? "Synthetic probability unavailable" : `Synthetic probability ${pctText} percent`
      }
    >
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--border)" strokeWidth={stroke} />
        <circle
          className="gauge-arc"
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeDasharray={`${dash} ${c - dash}`}
          strokeLinecap="round"
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          opacity={valid ? 1 : 0}
        />
      </svg>
      <div className="gauge-center" aria-hidden>
        <div className="gauge-value">
          {pctText == null ? "—" : pctText}
          {pctText != null && <small>%</small>}
        </div>
        <div className="gauge-caption">p(synthetic)</div>
      </div>
    </div>
  );
}
