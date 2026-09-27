import type { EvidenceEntry } from "@/lib/types";
import { clamp01, humanizeKey, isFiniteNumber, num } from "@/lib/format";

function suspicionColor(s: number) {
  if (s >= 0.66) return "var(--red)";
  if (s >= 0.33) return "var(--amber)";
  return "var(--green)";
}

export default function EvidenceCard({
  technique,
  entry,
}: {
  technique: string;
  entry: EvidenceEntry;
}) {
  const featureEntries = Object.entries(entry.features ?? {});
  const s = isFiniteNumber(entry.suspicion) ? clamp01(entry.suspicion) : null;
  return (
    <article className="evidence-card">
      <div className="evidence-card-head">
        <h4 className="evidence-name" style={{ margin: 0 }}>
          {humanizeKey(technique)}
        </h4>
        <span className={`badge ${entry.used_in_score ? "badge-used" : "badge-support"}`}>
          {entry.used_in_score ? "Used in score" : "Supporting evidence"}
        </span>
      </div>
      {entry.summary && <p className="evidence-summary">{entry.summary}</p>}

      {s != null && (
        <div className="meter-row">
          <span>Suspicion</span>
          <div
            className="suspicion-bar"
            role="meter"
            aria-label={`${humanizeKey(technique)} suspicion`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(s * 100)}
          >
            <div
              className="suspicion-bar-fill"
              style={{ transform: `scaleX(${s})`, background: suspicionColor(s) }}
            />
          </div>
          <span className="num">{Math.round(s * 100)}%</span>
        </div>
      )}

      {entry.flags && entry.flags.length > 0 && (
        <div className="flag-list" aria-label="Flags">
          {entry.flags.map((f, i) => (
            <span className="flag-pill" key={i}>
              {f}
            </span>
          ))}
        </div>
      )}

      {featureEntries.length > 0 && (
        <details className="disclosure">
          <summary>{featureEntries.length} features</summary>
          <table className="features-table">
            <tbody>
              {featureEntries.map(([k, v]) => (
                <tr key={k}>
                  <td>{k}</td>
                  <td>{num(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
    </article>
  );
}
