import type { Status } from "@/lib/types";

const LABELS: Record<Status, string> = {
  likely_human: "Likely human",
  inconclusive: "Inconclusive",
  likely_synthetic: "Likely synthetic",
  insufficient_evidence: "Insufficient evidence",
};

export default function StatusChip({ status }: { status: Status }) {
  return (
    <span className={`status-chip status-${status}`}>
      <span className="dot" />
      {LABELS[status] ?? status}
    </span>
  );
}
