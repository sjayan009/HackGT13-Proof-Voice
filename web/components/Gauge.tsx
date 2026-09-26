import type { Status } from "@/lib/types";

const STATUS_COLOR: Record<Status, string> = {
  likely_human: "var(--green)",
  inconclusive: "var(--amber)",
  likely_synthetic: "var(--red)",
  insufficient_evidence: "var(--gray)",
};

export default function Gauge({
  probability,
  status,
  label = "Synthetic probability",
}: {
  probability: number | null;
  status: Status | null;
  label?: string;
}) {
  const pct = probability == null ? null : Math.round(probability * 100);
  const size = 160;
  const stroke = 12;
  const r = (size - stroke) / 2;
  const circumference = 2 * Math.PI * r;
  const dash =
    pct == null ? 0 : (Math.max(0, Math.min(100, pct)) / 100) * circumference;
  const color = status ? STATUS_COLOR[status] : "var(--gray)";

  return (
    <div className="gauge-wrap">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="var(--border)"
          strokeWidth={stroke}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeDasharray={`${dash} ${circumference - dash}`}
          strokeLinecap="round"
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: "stroke-dasharray 0.3s ease" }}
        />
        <text
          x="50%"
          y="48%"
          textAnchor="middle"
          fontSize="30"
          fontWeight="700"
          fill="var(--text)"
          fontFamily="var(--mono)"
        >
          {pct == null ? "—" : `${pct}%`}
        </text>
        <text
          x="50%"
          y="64%"
          textAnchor="middle"
          fontSize="10"
          fill="var(--text-faint)"
        >
          p(synthetic)
        </text>
      </svg>
      <div className="gauge-label">{label}</div>
    </div>
  );
}
