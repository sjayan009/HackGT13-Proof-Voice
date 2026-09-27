"use client";

import { useEffect, useRef, useState } from "react";
import type { AnalysisReport, ExplainResponse } from "@/lib/types";
import { ApiError, describeError, explainReport, isAbort } from "@/lib/api";
import {
  STATUS_LABEL,
  downloadJson,
  isFiniteNumber,
  ms,
  pct,
  reportFilename,
  verdictSentence,
} from "@/lib/format";
import EvidenceScale, { LeaningValue } from "./EvidenceScale";
import StatusChip from "./StatusChip";
import TrustTimelineChart from "./TrustTimelineChart";
import EvidenceCard from "./EvidenceCard";
import BranchLogTable from "./BranchLogTable";
import FileMetadataPanel from "./FileMetadataPanel";
import Disclaimer from "./Disclaimer";
import Callout from "./Callout";
import Icon from "./Icon";
import { useAudioPlayhead } from "@/lib/useAudioPlayhead";

export interface GroundTruth {
  truth: "bonafide" | "spoof";
  label: string;
}

function TruthVerdict({ truth, status }: { truth: GroundTruth; status: AnalysisReport["status"] }) {
  const isFake = truth.truth === "spoof";
  let tone: "ok" | "bad" | "neutral";
  let text: string;
  if (status === "inconclusive" || status === "insufficient_evidence") {
    tone = "neutral";
    text = "Not decided: would be routed to a human analyst";
  } else if ((status === "likely_synthetic") === isFake) {
    tone = "ok";
    text = "Detector was correct";
  } else {
    tone = "bad";
    text = isFake ? "Missed: the detector called this synthetic clip human" : "False alarm on real speech";
  }
  return (
    <div className={`truth-verdict truth-${tone}`} role="status">
      <Icon name={tone === "ok" ? "check" : tone === "bad" ? "x" : "info"} size={15} strokeWidth={2.2} />
      <span>
        Ground truth: <strong>{isFake ? "Synthetic" : "Real"}</strong>
        <span className="truth-sep"> · </span>
        {text}
      </span>
    </div>
  );
}

function fmtClock(ms: number | null | undefined) {
  if (!isFiniteNumber(ms)) return "0:00";
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

type ExplainState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ok"; data: ExplainResponse }
  | { kind: "unavailable"; message: string }
  | { kind: "error"; message: string };

function ExplainPanel({ report }: { report: AnalysisReport }) {
  const [state, setState] = useState<ExplainState>({ kind: "idle" });
  const ctrl = useRef<AbortController | null>(null);

  useEffect(() => () => ctrl.current?.abort(), []);
  useEffect(() => {
    ctrl.current?.abort();
    setState({ kind: "idle" });
  }, [report.id]);

  const run = async () => {
    ctrl.current?.abort();
    const c = new AbortController();
    ctrl.current = c;
    setState({ kind: "loading" });
    try {
      const data = await explainReport(report, c.signal);
      if (!c.signal.aborted) setState({ kind: "ok", data });
    } catch (err) {
      if (isAbort(err) || c.signal.aborted) return;
      if (err instanceof ApiError && err.status === 503) {
        setState({ kind: "unavailable", message: err.message });
      } else {
        setState({ kind: "error", message: describeError(err) });
      }
    }
  };

  return (
    <section className="panel" aria-labelledby={`explain-${report.id}`}>
      <div className="panel-head">
        <h3 id={`explain-${report.id}`}>Plain-language explanation</h3>
        {state.kind === "ok" && (
          <span className="panel-sub">
            {state.data.model}
            {state.data.grounded ? " · grounded in report fields" : " · not verified as grounded"}
          </span>
        )}
      </div>
      {state.kind === "idle" && (
        <div className="control-bar">
          <span className="panel-sub" style={{ maxWidth: "60ch" }}>
            Ask Grok to restate this report in plain English. It may only use fields already in the report and
            does not change the score.
          </span>
          <button className="btn" onClick={run}>
            <Icon name="wand" size={15} />
            Explain this result
          </button>
        </div>
      )}
      {state.kind === "loading" && (
        <div aria-busy="true" aria-live="polite">
          <div className="skeleton" style={{ height: 12, width: "92%" }} />
          <div className="skeleton" style={{ height: 12, width: "84%", marginTop: 8 }} />
          <div className="skeleton" style={{ height: 12, width: "60%", marginTop: 8 }} />
        </div>
      )}
      {state.kind === "ok" && (
        <p className="explain-text appear" aria-live="polite">
          {state.data.text}
        </p>
      )}
      {state.kind === "unavailable" && (
        <Callout tone="info" title="Explanations are unavailable">
          The backend has no xAI key configured ({state.message}). The report above is complete without it.
        </Callout>
      )}
      {state.kind === "error" && (
        <Callout
          tone="error"
          title="Could not generate an explanation"
          action={
            <button className="btn btn-sm" onClick={run}>
              Retry
            </button>
          }
        >
          {state.message}
        </Callout>
      )}
    </section>
  );
}

export default function ReportView({
  report,
  waveformPeaks,
  headerExtra,
  audioUrl,
  groundTruth,
  precomputedNote,
}: {
  report: AnalysisReport;
  waveformPeaks?: number[] | null;
  headerExtra?: React.ReactNode;
  /** Local/object URL of the analyzed audio, enabling listen-along on the timeline. */
  audioUrl?: string | null;
  groundTruth?: GroundTruth | null;
  /** Set when this report was loaded from the precomputed sample library instead of a live backend. */
  precomputedNote?: string | null;
}) {
  const player = useAudioPlayhead(audioUrl, `report-${report.id}`);
  // Scoring branches first, then by suspicion (highest first).
  const evidenceEntries = Object.entries(report.evidence ?? {}).sort(([, a], [, b]) => {
    if (a.used_in_score !== b.used_in_score) return a.used_in_score ? -1 : 1;
    return (b.suspicion ?? -1) - (a.suspicion ?? -1);
  });
  const whyRun = Object.entries(report.why_run ?? {});
  const durationMs = isFiniteNumber(report.file?.duration_s) ? report.file.duration_s * 1000 : null;

  return (
    <div className="stack appear">
      {precomputedNote && (
        <Callout tone="warn" title="Showing a precomputed result">
          {precomputedNote}
        </Callout>
      )}
      <section className="panel" aria-label="Verdict">
        <div className="verdict">
          <div className="verdict-reading">
            <StatusChip status={report.status} />
            <LeaningValue probability={report.synthetic_probability} status={report.status} />
          </div>
          <div className="verdict-body">
            <h2 className="verdict-headline">
              {STATUS_LABEL[report.status] ?? report.status}
              {report.file?.name ? (
                <span style={{ color: "var(--text-faint)", fontWeight: 500 }}> · {report.file.name}</span>
              ) : null}
            </h2>
            <p className="verdict-sentence">{verdictSentence(report)}</p>
            {groundTruth && <TruthVerdict truth={groundTruth} status={report.status} />}
            <div className="verdict-actions">
              <button className="btn btn-sm" onClick={() => downloadJson(report, reportFilename(report))}>
                <Icon name="download" size={14} />
                Export JSON
              </button>
              {headerExtra}
            </div>
          </div>
        </div>
        <EvidenceScale
          probability={report.synthetic_probability}
          threshold={report.decision_threshold}
          status={report.status}
        />
        <div className="stat-grid">
          <div className="stat">
            <div className="stat-label">p(synthetic)</div>
            <div className="stat-value">
              {isFiniteNumber(report.synthetic_probability) ? report.synthetic_probability.toFixed(4) : "—"}
            </div>
          </div>
          <div className="stat">
            <div className="stat-label">Analysis confidence</div>
            <div className="stat-value">{pct(report.analysis_confidence)}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Decision threshold</div>
            <div className="stat-value">
              {isFiniteNumber(report.decision_threshold) ? report.decision_threshold.toFixed(3) : "—"}
            </div>
          </div>
          <div className="stat">
            <div className="stat-label">cm_score</div>
            <div className="stat-value">{isFiniteNumber(report.cm_score) ? report.cm_score.toFixed(4) : "—"}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Time to confidence</div>
            <div className="stat-value">{ms(report.time_to_confidence_ms)}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Processing time</div>
            <div className="stat-value">{ms(report.processing_ms)}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Detector</div>
            <div
              className="stat-value"
              style={{ fontFamily: "var(--sans)", fontSize: "0.8125rem" }}
              title={`${report.detector?.name ?? ""} ${report.detector?.version ?? ""}`}
            >
              {report.detector?.name ?? "—"} {report.detector?.version}
            </div>
          </div>
        </div>
      </section>

      <section className="panel" aria-labelledby={`tl-${report.id}`}>
        <div className="panel-head">
          <div className="control-group" style={{ gap: 10 }}>
            <h3 id={`tl-${report.id}`}>Trust timeline</h3>
            {player.available && (
              <button
                className="listen-btn"
                onClick={player.toggle}
                aria-label={player.playing ? "Pause audio" : "Play audio along the timeline"}
                aria-pressed={player.playing}
              >
                <Icon name={player.playing ? "pause" : "play"} size={12} />
                <span className="num">
                  {fmtClock(player.positionMs)} / {fmtClock(player.durationMs ?? durationMs)}
                </span>
              </button>
            )}
          </div>
          <span className="panel-sub">
            {report.timeline?.length ?? 0} windows ·{" "}
            {report.suspicious_regions?.length
              ? `${report.suspicious_regions.length} suspicious region${
                  report.suspicious_regions.length === 1 ? "" : "s"
                }`
              : "no suspicious regions"}
          </span>
        </div>
        <TrustTimelineChart
          timeline={report.timeline}
          suspiciousRegions={report.suspicious_regions}
          decisionThreshold={report.decision_threshold}
          waveformPeaks={waveformPeaks}
          durationMs={durationMs}
          playheadMs={player.positionMs}
          onSeek={player.available ? player.seek : undefined}
        />
        {player.available && (
          <p className="panel-sub" style={{ margin: "8px 0 0" }}>
            Click anywhere on the chart to listen from that moment.
          </p>
        )}
      </section>

      <div className="grid-2" style={{ alignItems: "start" }}>
        <section className="panel" aria-labelledby={`ev-${report.id}`}>
          <div className="panel-head">
            <h3 id={`ev-${report.id}`}>Evidence by technique</h3>
            <span className="panel-sub">{evidenceEntries.length} techniques</span>
          </div>
          {evidenceEntries.length === 0 ? (
            <div className="empty-state">No evidence entries reported.</div>
          ) : (
            <div className="evidence-list">
              {evidenceEntries.map(([technique, entry]) => (
                <EvidenceCard key={technique} technique={technique} entry={entry} />
              ))}
            </div>
          )}
          {whyRun.length > 0 && (
            <details className="disclosure" style={{ marginTop: 14 }}>
              <summary>Why extra techniques ran</summary>
              <ul className="notes-list" style={{ marginTop: 8 }}>
                {whyRun.map(([k, v]) => (
                  <li key={k}>
                    <strong style={{ color: "var(--text)" }}>{k}</strong>: {v}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </section>

        <div className="stack">
          <section className="panel" aria-labelledby={`meta-${report.id}`}>
            <div className="panel-head">
              <h3 id={`meta-${report.id}`}>File metadata</h3>
            </div>
            <FileMetadataPanel file={report.file} />
          </section>
          <section className="panel" aria-labelledby={`bl-${report.id}`}>
            <div className="panel-head">
              <h3 id={`bl-${report.id}`}>Branch execution log</h3>
              <span className="panel-sub">deterministic routing</span>
            </div>
            <BranchLogTable log={report.branch_log} />
          </section>
        </div>
      </div>

      <ExplainPanel report={report} />

      <Disclaimer text={report.disclaimer} />
    </div>
  );
}
