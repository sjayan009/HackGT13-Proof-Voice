"use client";

import { useEffect, useState } from "react";
import type { Status } from "@/lib/types";
import { STATUS_COLOR, clamp01, decisionBands, isFiniteNumber, leaning } from "@/lib/format";

/** Headline reading in the direction of the evidence: "99.7% human" / "98% synthetic". */
export function LeaningValue({
  probability,
  status,
  size = "lg",
}: {
  probability: number | null;
  status: Status | null;
  size?: "lg" | "md";
}) {
  const l = leaning(probability);
  const color = status ? STATUS_COLOR[status] : "var(--text)";
  return (
    <div className={`leaning leaning-${size}`}>
      <span className="leaning-value num" style={{ color }}>
        {l ? l.text : "—"}
      </span>
      {l && <span className="leaning-side">{l.side}</span>}
    </div>
  );
}

/**
 * Human ↔ Synthetic scale with the backend's three decision zones and a marker at p(synthetic).
 * Makes it visible why, e.g., "80% human" can still be inconclusive.
 */
export default function EvidenceScale({
  probability,
  threshold,
  status,
}: {
  probability: number | null;
  threshold: number | null;
  status: Status | null;
}) {
  const valid = isFiniteNumber(probability);
  const p = valid ? clamp01(probability) : 0.5;
  // Start at the center and settle on the reading, so the marker visibly travels toward its side.
  const [shown, setShown] = useState(0.5);
  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(p));
    return () => cancelAnimationFrame(id);
  }, [p]);

  const bands = isFiniteNumber(threshold) ? decisionBands(threshold) : null;
  const pct = (v: number) => `${(v * 100).toFixed(2)}%`;

  return (
    <div className="escale">
      <div
        className="escale-track"
        role="meter"
        aria-label="Synthetic probability on a human-to-synthetic scale"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={valid ? Math.round(p * 100) : undefined}
        aria-valuetext={valid ? `${leaning(p)!.text} ${leaning(p)!.side}` : "unavailable"}
      >
        {bands ? (
          <>
            <span className="escale-zone" style={{ left: 0, width: pct(bands.human), background: "var(--green)" }} />
            <span
              className="escale-zone"
              style={{ left: pct(bands.human), width: pct(bands.synthetic - bands.human), background: "var(--amber)" }}
            />
            <span
              className="escale-zone"
              style={{ left: pct(bands.synthetic), right: 0, background: "var(--red)" }}
            />
            <span className="escale-tick" style={{ left: pct(threshold!) }} title={`Decision threshold ${threshold!.toFixed(3)}`} />
          </>
        ) : (
          <span className="escale-zone" style={{ left: 0, right: 0, background: "var(--gray)" }} />
        )}
        {valid && (
          <span
            className="escale-marker"
            style={{
              left: `${shown * 100}%`,
              borderColor: status ? STATUS_COLOR[status] : "var(--text)",
            }}
          />
        )}
      </div>
      <div className="escale-labels">
        <span>Human</span>
        {bands && (
          <span className="escale-mid" style={{ left: pct((bands.human + bands.synthetic) / 2) }}>
            Inconclusive
          </span>
        )}
        <span>Synthetic</span>
      </div>
    </div>
  );
}
