import type { AnalysisReport } from "@/lib/types";
import Gauge from "./Gauge";
import StatusChip from "./StatusChip";
import TrustTimelineChart from "./TrustTimelineChart";
import Waveform from "./Waveform";
import EvidenceCard from "./EvidenceCard";
import BranchLogTable from "./BranchLogTable";
import FileMetadataPanel from "./FileMetadataPanel";
import Disclaimer from "./Disclaimer";

export default function ReportView({
  report,
  waveformPeaks,
}: {
  report: AnalysisReport;
  waveformPeaks?: number[] | null;
}) {
  const evidenceEntries = Object.entries(report.evidence ?? {});

  return (
    <div>
      <div className="panel">
        <div className="grid-2">
          <div style={{ display: "flex", alignItems: "center", gap: 24 }}>
            <Gauge
              probability={report.synthetic_probability}
              status={report.status}
            />
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <StatusChip status={report.status} />
              <div className="kv-list">
                <div className="k">Analysis confidence</div>
                <div className="v">
                  {(report.analysis_confidence * 100).toFixed(0)}%
                </div>
                <div className="k">Decision threshold</div>
                <div className="v">{report.decision_threshold.toFixed(2)}</div>
                <div className="k">cm_score</div>
                <div className="v">{report.cm_score.toFixed(4)}</div>
                <div className="k">Time to confidence</div>
                <div className="v">
                  {report.time_to_confidence_ms != null
                    ? `${report.time_to_confidence_ms} ms`
                    : "n/a"}
                </div>
                <div className="k">Detector</div>
                <div className="v">
                  {report.detector?.name} {report.detector?.version}
                </div>
                <div className="k">Processing time</div>
                <div className="v">{report.processing_ms.toFixed(0)} ms</div>
              </div>
            </div>
          </div>
          <div>
            <h2 style={{ marginTop: 0 }}>File metadata</h2>
            <FileMetadataPanel file={report.file} />
          </div>
        </div>
      </div>

      {waveformPeaks && (
        <div className="panel">
          <h2>Waveform</h2>
          <Waveform peaks={waveformPeaks} />
        </div>
      )}

      <div className="panel">
        <h2>Trust timeline</h2>
        <TrustTimelineChart
          timeline={report.timeline}
          suspiciousRegions={report.suspicious_regions}
          decisionThreshold={report.decision_threshold}
        />
        {report.suspicious_regions?.length > 0 && (
          <div style={{ marginTop: 10, fontSize: 12, color: "var(--text-dim)" }}>
            {report.suspicious_regions.length} suspicious region(s) flagged
            (shaded red).
          </div>
        )}
      </div>

      <div className="panel">
        <h2>Evidence by technique</h2>
        {evidenceEntries.length === 0 ? (
          <div className="empty-state">No evidence entries reported.</div>
        ) : (
          evidenceEntries.map(([technique, entry]) => (
            <EvidenceCard key={technique} technique={technique} entry={entry} />
          ))
        )}
        {report.why_run && Object.keys(report.why_run).length > 0 && (
          <details className="features" style={{ marginTop: 12 }}>
            <summary>Why extra techniques ran</summary>
            <ul style={{ fontSize: 12, color: "var(--text-dim)" }}>
              {Object.entries(report.why_run).map(([k, v]) => (
                <li key={k}>
                  <strong>{k}</strong>: {v}
                </li>
              ))}
            </ul>
          </details>
        )}
      </div>

      <div className="panel">
        <h2>Branch execution log</h2>
        <BranchLogTable log={report.branch_log} />
      </div>

      <Disclaimer text={report.disclaimer} />
    </div>
  );
}
