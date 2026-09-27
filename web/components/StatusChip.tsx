import type { Status } from "@/lib/types";
import { STATUS_LABEL } from "@/lib/format";

export default function StatusChip({ status }: { status: Status }) {
  return (
    <span className={`status-chip status-${status}`}>
      <span className="dot" aria-hidden />
      {STATUS_LABEL[status] ?? status}
    </span>
  );
}
