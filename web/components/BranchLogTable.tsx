import type { BranchLogEntry } from "@/lib/types";

export default function BranchLogTable({ log }: { log: BranchLogEntry[] }) {
  if (!log || log.length === 0) {
    return <div className="empty-state">No branch execution log reported.</div>;
  }
  return (
    <table className="data-table">
      <thead>
        <tr>
          <th>Technique</th>
          <th>Ran</th>
          <th>Reason</th>
          <th>Runtime (ms)</th>
        </tr>
      </thead>
      <tbody>
        {log.map((entry, i) => (
          <tr key={i} className={entry.ran ? "" : "ran-false"}>
            <td>{entry.technique}</td>
            <td className={entry.ran ? "ok-yes" : "ok-no"}>
              {entry.ran ? "yes" : "no"}
            </td>
            <td>{entry.reason}</td>
            <td className="mono">{entry.runtime_ms?.toFixed?.(1) ?? "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
