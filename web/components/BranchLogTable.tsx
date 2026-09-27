import type { BranchLogEntry } from "@/lib/types";
import { isFiniteNumber } from "@/lib/format";

export default function BranchLogTable({ log }: { log: BranchLogEntry[] }) {
  if (!log || log.length === 0) {
    return <div className="empty-state">No branch execution log reported.</div>;
  }
  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>
            <th scope="col">Technique</th>
            <th scope="col">Ran</th>
            <th scope="col">Reason</th>
            <th scope="col" className="num">
              Runtime
            </th>
          </tr>
        </thead>
        <tbody>
          {log.map((entry, i) => (
            <tr key={`${entry.technique}-${i}`} className={entry.ran ? "" : "ran-false"}>
              <td style={{ fontWeight: 500, whiteSpace: "nowrap" }}>{entry.technique}</td>
              <td>
                <span className={`ran-dot ${entry.ran ? "ok-yes" : "ok-no"}`}>
                  {entry.ran ? "Ran" : "Skipped"}
                </span>
              </td>
              <td style={{ color: "var(--text-dim)" }}>{entry.reason}</td>
              <td className="num mono">
                {isFiniteNumber(entry.runtime_ms) ? `${entry.runtime_ms.toFixed(1)} ms` : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
