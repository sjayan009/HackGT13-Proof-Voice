"use client";

import { useCallback, useRef, useState } from "react";
import type {
  AnalysisReport,
  AnalysisWindowEvent,
  RedteamSource,
  RedteamStreamServerEvent,
} from "@/lib/types";
import { redteamGrok, redteamStreamUrl, ApiError } from "@/lib/api";
import { base64ToInt16Array, PcmStreamPlayer } from "@/lib/audio";
import RollingProbabilityChart from "./RollingProbabilityChart";
import ReportView from "./ReportView";
import StatusChip from "./StatusChip";

const SOURCE_LABEL: Record<RedteamSource, string> = {
  grok_voice: "Grok Voice (realtime)",
  grok_tts: "Grok streaming TTS (fallback)",
  cached_fixture: "Cached fixture (offline fallback)",
};

export default function RedTeam({
  lastHumanReport,
}: {
  lastHumanReport?: AnalysisReport | null;
}) {
  const [text, setText] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [events, setEvents] = useState<AnalysisWindowEvent[]>([]);
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [source, setSource] = useState<RedteamSource | null>(null);
  const [mode, setMode] = useState<"stream" | "fallback" | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const playerRef = useRef<PcmStreamPlayer | null>(null);

  const reset = () => {
    setEvents([]);
    setReport(null);
    setSource(null);
    setError(null);
  };

  const runStream = useCallback(() => {
    reset();
    setMode("stream");
    setRunning(true);

    let ws: WebSocket;
    try {
      ws = new WebSocket(redteamStreamUrl());
    } catch {
      setError("Could not open WebSocket connection to the backend.");
      setRunning(false);
      return;
    }
    wsRef.current = ws;

    ws.onerror = () => {
      setError(
        "Red-team stream error — is the backend running at the configured NEXT_PUBLIC_WS_BASE_URL?"
      );
      setRunning(false);
    };

    ws.onopen = () => {
      ws.send(JSON.stringify({ type: "start", text: text || undefined }));
    };

    ws.onmessage = (ev) => {
      try {
        const data: RedteamStreamServerEvent = JSON.parse(ev.data);
        if (data.type === "audio.chunk") {
          if (!playerRef.current) {
            playerRef.current = new PcmStreamPlayer(data.sample_rate);
          }
          const int16 = base64ToInt16Array(data.pcm16_b64);
          playerRef.current.enqueue(int16);
        } else if (data.type === "analysis.window") {
          setEvents((prev) => [...prev, data]);
        } else if (data.type === "redteam.done") {
          setReport(data.report);
          setSource(data.source);
          setRunning(false);
          ws.close();
        } else if (data.type === "error") {
          setError(data.detail);
          setRunning(false);
        }
      } catch {
        // ignore malformed frames
      }
    };

    ws.onclose = () => {
      setRunning(false);
    };
  }, [text]);

  const runFallback = useCallback(async () => {
    reset();
    setMode("fallback");
    setRunning(true);
    try {
      const result = await redteamGrok({ text: text || undefined });
      setReport(result.report);
      setSource(result.source);
    } catch (err) {
      if (err instanceof ApiError) setError(err.message);
      else if (err instanceof TypeError)
        setError(
          "Could not reach the ProofVoice API. Is the backend running at the configured NEXT_PUBLIC_API_BASE_URL?"
        );
      else setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setRunning(false);
    }
  }, [text]);

  const latest = events[events.length - 1];

  return (
    <div>
      <div className="panel">
        <h2>Generate adversarial sample</h2>
        <div className="field-row">
          <input
            type="text"
            placeholder="Optional line for Grok to speak…"
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={running}
          />
          <button className="btn btn-primary" onClick={runStream} disabled={running}>
            Generate with Grok Voice &amp; analyze
          </button>
          <button className="btn" onClick={runFallback} disabled={running}>
            Fallback: non-streaming /redteam/grok
          </button>
        </div>
        {error && (
          <div className="error-state" style={{ marginTop: 12 }}>
            {error}
          </div>
        )}
        {source && (
          <div style={{ marginTop: 12, fontSize: 13 }}>
            Source: <strong>{SOURCE_LABEL[source]}</strong>{" "}
            <span style={{ color: "var(--text-faint)" }}>({source})</span>
          </div>
        )}
      </div>

      {mode === "stream" && (running || events.length > 0) && !report && (
        <div className="panel">
          <h2>Live analysis of generated audio</h2>
          <RollingProbabilityChart events={events} />
          {latest && (
            <div style={{ marginTop: 10 }}>
              <StatusChip status={latest.status} />
            </div>
          )}
        </div>
      )}

      {report && (
        <div style={{ marginTop: 16 }}>
          {lastHumanReport ? (
            <div className="grid-2">
              <div>
                <h2 style={{ fontSize: 13, color: "var(--text-dim)", textTransform: "uppercase" }}>
                  Last human sample (Live Trust)
                </h2>
                <div className="panel">
                  <StatusChip status={lastHumanReport.status} />
                  <div className="kv-list" style={{ marginTop: 10 }}>
                    <div className="k">Synthetic probability</div>
                    <div className="v">
                      {(lastHumanReport.synthetic_probability * 100).toFixed(0)}%
                    </div>
                    <div className="k">Analysis confidence</div>
                    <div className="v">
                      {(lastHumanReport.analysis_confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
              </div>
              <div>
                <h2 style={{ fontSize: 13, color: "var(--text-dim)", textTransform: "uppercase" }}>
                  Grok sample ({source})
                </h2>
                <div className="panel">
                  <StatusChip status={report.status} />
                  <div className="kv-list" style={{ marginTop: 10 }}>
                    <div className="k">Synthetic probability</div>
                    <div className="v">
                      {(report.synthetic_probability * 100).toFixed(0)}%
                    </div>
                    <div className="k">Analysis confidence</div>
                    <div className="v">{(report.analysis_confidence * 100).toFixed(0)}%</div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="empty-state" style={{ marginBottom: 16 }}>
              No Live Trust session result in memory yet — run Live Trust first
              to see a human-vs-Grok comparison.
            </div>
          )}
          <div style={{ marginTop: 16 }}>
            <ReportView report={report} />
          </div>
        </div>
      )}
    </div>
  );
}
