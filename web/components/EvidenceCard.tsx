import type { EvidenceEntry } from "@/lib/types";

export default function EvidenceCard({
  technique,
  entry,
}: {
  technique: string;
  entry: EvidenceEntry;
}) {
  const featureEntries = Object.entries(entry.features ?? {});
  return (
    <div className="evidence-card">
      <div className="evidence-card-head">
        <span className="evidence-name">{technique.replace(/_/g, " ")}</span>
        <span className={`badge ${entry.used_in_score ? "badge-used" : "badge-support"}`}>
          {entry.used_in_score ? "Used in score" : "Supporting evidence"}
        </span>
      </div>
      <div className="evidence-summary">{entry.summary}</div>

      {entry.suspicion != null && (
        <div>
          <div style={{ fontSize: 11, color: "var(--text-faint)" }}>
            Suspicion: {(entry.suspicion * 100).toFixed(0)}%
          </div>
          <div className="suspicion-bar">
            <div
              className="suspicion-bar-fill"
              style={{ width: `${Math.max(0, Math.min(100, entry.suspicion * 100))}%` }}
            />
          </div>
        </div>
      )}

      {entry.flags && entry.flags.length > 0 && (
        <div className="flag-list">
          {entry.flags.map((f, i) => (
            <span className="flag-pill" key={i}>
              {f}
            </span>
          ))}
        </div>
      )}

      {featureEntries.length > 0 && (
        <details className="features">
          <summary>Features ({featureEntries.length})</summary>
          <table className="features-table">
            <tbody>
              {featureEntries.map(([k, v]) => (
                <tr key={k}>
                  <td>{k}</td>
                  <td>{typeof v === "number" ? v.toFixed(4) : String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
    </div>
  );
}
