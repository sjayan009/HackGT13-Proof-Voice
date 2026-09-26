"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  AnalysisReport,
  AnalysisWindowEvent,
  StreamServerEvent,
} from "@/lib/types";
import { analyzeStreamUrl } from "@/lib/api";
import { startMicCapture, type MicCaptureHandle } from "@/lib/mic";
import StatusChip from "./StatusChip";
import RollingProbabilityChart from "./RollingProbabilityChart";
import LevelMeter from "./LevelMeter";
import ReportView from "./ReportView";

type SessionState = "idle" | "connecting" | "live" | "stopping" | "error";

export default function LiveTrust({
  onReportReady,
}: {
  onReportReady?: (report: AnalysisReport) => void;
}) {
  const [state, setState] = useState<SessionState>("idle");
  const [events, setEvents] = useState<AnalysisWindowEvent[]>([]);
  const [finalReport, setFinalReport] = useState<AnalysisReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [levelDbfs, setLevelDbfs] = useState<number | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const micRef = useRef<MicCaptureHandle | null>(null);

  const cleanup = useCallback(() => {
    try {
      micRef.current?.stop();
    } catch {}
    micRef.current = null;
    try {
      wsRef.current?.close();
    } catch {}
    wsRef.current = null;
  }, []);

  useEffect(() => () => cleanup(), [cleanup]);

  const start = useCallback(async () => {
    setError(null);
    setEvents([]);
    setFinalReport(null);
    setState("connecting");

    if (!navigator.mediaDevices?.getUserMedia) {
      setError("This browser does not support microphone capture (getUserMedia).");
      setState("error");
      return;
    }
    if (typeof WebSocket === "undefined") {
      setError("This browser does not support WebSocket streaming.");
      setState("error");
      return;
    }

    let ws: WebSocket;
    try {
      ws = new WebSocket(analyzeStreamUrl());
    } catch (err) {
      setError("Could not open WebSocket connection to the backend.");
      setState("error");
      return;
    }
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onerror = () => {
      setError(
        "WebSocket error — is the backend running at the configured NEXT_PUBLIC_WS_BASE_URL?"
      );
      setState("error");
      cleanup();
    };

    ws.onclose = () => {
      setState((s) => (s === "stopping" ? "idle" : s === "live" ? "error" : s));
    };

    ws.onmessage = (ev) => {
      try {
        const data: StreamServerEvent = JSON.parse(ev.data);
        if (data.type === "ready") {
          // ready to receive PCM — nothing to render
        } else if (data.type === "analysis.window") {
          setEvents((prev) => [...prev, data]);
        } else if (data.type === "analysis.final") {
          setFinalReport(data.report);
          onReportReady?.(data.report);
          setState("idle");
          cleanup();
        } else if (data.type === "error") {
          setError(data.detail);
          setState("error");
        }
      } catch {
        // ignore malformed frames
      }
    };

    ws.onopen = async () => {
      try {
        const mic = await startMicCapture({
          onChunk: (chunk) => {
            if (ws.readyState === WebSocket.OPEN) {
              ws.send(chunk.buffer);
            }
          },
          onLevel: (dbfs) => setLevelDbfs(dbfs),
        });
        micRef.current = mic;
        ws.send(
          JSON.stringify({
            type: "start",
            sample_rate: mic.sampleRate,
            encoding: "f32le",
            source: "mic",
          })
        );
        setState("live");
      } catch (err) {
        setError(
          err instanceof Error
            ? `Microphone access failed: ${err.message}`
            : "Microphone access failed."
        );
        setState("error");
        cleanup();
      }
    };
  }, [cleanup, onReportReady]);

  const stop = useCallback(() => {
    setState("stopping");
    try {
      micRef.current?.stop();
    } catch {}
    micRef.current = null;
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "stop" }));
    } else {
      setState("idle");
    }
  }, []);

  const latest = events[events.length - 1];

  return (
    <div>
      <div className="panel">
        <div className="field-row" style={{ justifyContent: "space-between" }}>
          <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
            {state === "idle" || state === "error" ? (
              <button className="btn btn-primary" onClick={start}>
                Start listening
              </button>
            ) : (
              <button
                className="btn btn-danger"
                onClick={stop}
                disabled={state === "stopping"}
              >
                {state === "stopping" ? "Stopping…" : "Stop"}
              </button>
            )}
            <span style={{ fontSize: 12, color: "var(--text-dim)" }}>
              {state === "connecting" && "Connecting…"}
              {state === "live" && "Live"}
              {state === "idle" && !finalReport && "Idle — press start"}
              {state === "stopping" && "Finalizing report…"}
            </span>
          </div>
          {latest && <StatusChip status={latest.status} />}
        </div>

        {error && (
          <div className="error-state" style={{ marginTop: 12 }}>
            {error}
          </div>
        )}

        {state === "live" && (
          <div style={{ marginTop: 14 }}>
            <LevelMeter dbfs={levelDbfs} />
          </div>
        )}
      </div>

      {(state === "live" || state === "stopping" || (events.length > 0 && !finalReport)) && (
        <div className="panel">
          <h2>Rolling probability</h2>
          <RollingProbabilityChart events={events} />
          <div className="grid-3" style={{ marginTop: 14 }}>
            <div>
              <div className="k" style={{ color: "var(--text-dim)", fontSize: 12 }}>
                Rolling probability
              </div>
              <div className="mono" style={{ fontSize: 20 }}>
                {latest ? `${(latest.rolling_probability * 100).toFixed(0)}%` : "—"}
              </div>
            </div>
            <div>
              <div className="k" style={{ color: "var(--text-dim)", fontSize: 12 }}>
                Analysis confidence
              </div>
              <div className="mono" style={{ fontSize: 20 }}>
                {latest ? `${(latest.analysis_confidence * 100).toFixed(0)}%` : "—"}
              </div>
            </div>
            <div>
              <div className="k" style={{ color: "var(--text-dim)", fontSize: 12 }}>
                Time to confidence
              </div>
              <div className="mono" style={{ fontSize: 20 }}>
                {latest?.time_to_confidence_ms != null
                  ? `${latest.time_to_confidence_ms} ms`
                  : "—"}
              </div>
            </div>
          </div>
        </div>
      )}

      {finalReport && (
        <div style={{ marginTop: 16 }}>
          <h2 style={{ fontSize: 14, color: "var(--text-dim)", textTransform: "uppercase" }}>
            Final report
          </h2>
          <ReportView report={finalReport} />
        </div>
      )}
    </div>
  );
}
