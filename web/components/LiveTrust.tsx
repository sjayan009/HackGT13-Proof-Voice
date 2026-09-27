"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { AnalysisReport, AnalysisWindowEvent, StreamServerEvent } from "@/lib/types";
import { analyzeStreamUrl, wsBase } from "@/lib/api";
import { startClipCapture, startMicCapture, type MicCaptureHandle } from "@/lib/mic";
import { loadSampleManifest, type Sample } from "@/lib/samples";
import { leaning, ms, pct } from "@/lib/format";
import StatusChip from "./StatusChip";
import RollingProbabilityChart from "./RollingProbabilityChart";
import LevelMeter from "./LevelMeter";
import ReportView from "./ReportView";
import Callout from "./Callout";
import Icon from "./Icon";

type SessionState = "idle" | "connecting" | "live" | "stopping" | "error";
type InputMode = "mic" | "clip";
interface Clip {
  name: string;
  blob: Blob;
  sampleId?: string;
}

const FINALIZE_TIMEOUT_MS = 20_000;

function micErrorMessage(err: unknown): string {
  if (err instanceof DOMException) {
    if (err.name === "NotAllowedError" || err.name === "SecurityError")
      return "Microphone permission was denied. Allow microphone access for this site in your browser settings, then try again.";
    if (err.name === "NotFoundError") return "No microphone was found on this device.";
    if (err.name === "NotReadableError") return "The microphone is in use by another application.";
  }
  return `Microphone access failed: ${err instanceof Error ? err.message : String(err)}`;
}

function clock(msValue: number) {
  const s = Math.floor(msValue / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

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
  const [elapsed, setElapsed] = useState(0);
  const [mode, setMode] = useState<InputMode>("mic");
  const [clip, setClip] = useState<Clip | null>(null);
  const [clipProgress, setClipProgress] = useState<{ t: number; d: number } | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const clipInputRef = useRef<HTMLInputElement>(null);
  const stopRef = useRef<() => void>(() => {});
  const sessionModeRef = useRef<InputMode>("mic");

  useEffect(() => {
    loadSampleManifest().then((m) => setSamples(m?.samples ?? []));
  }, []);

  const pickSample = async (smp: Sample) => {
    try {
      const res = await fetch(smp.file);
      if (!res.ok) throw new Error(String(res.status));
      setClip({ name: smp.title, blob: await res.blob(), sampleId: smp.id });
      setError(null);
    } catch {
      setError("Could not load that sample.");
    }
  };

  const wsRef = useRef<WebSocket | null>(null);
  const micRef = useRef<MicCaptureHandle | null>(null);
  const stateRef = useRef<SessionState>("idle");
  const finalizeTimer = useRef<ReturnType<typeof setTimeout>>();

  const setSession = (s: SessionState) => {
    stateRef.current = s;
    setState(s);
  };

  const cleanup = useCallback(() => {
    clearTimeout(finalizeTimer.current);
    try {
      micRef.current?.stop();
    } catch {}
    micRef.current = null;
    const ws = wsRef.current;
    wsRef.current = null;
    if (ws) {
      ws.onclose = ws.onerror = ws.onmessage = ws.onopen = null;
      try {
        ws.close();
      } catch {}
    }
    setLevelDbfs(null);
  }, []);

  useEffect(() => () => cleanup(), [cleanup]);

  useEffect(() => {
    if (state !== "live") return;
    const t0 = performance.now();
    setElapsed(0);
    const id = setInterval(() => setElapsed(performance.now() - t0), 250);
    return () => clearInterval(id);
  }, [state]);

  const fail = useCallback(
    (message: string) => {
      setError(message);
      setSession("error");
      cleanup();
    },
    [cleanup]
  );

  const start = useCallback(async () => {
    cleanup();
    setError(null);
    setEvents([]);
    setFinalReport(null);
    setSession("connecting");

    const sessionMode = mode;
    const sessionClip = clip;
    sessionModeRef.current = sessionMode;
    setClipProgress(null);
    if (sessionMode === "clip" && !sessionClip) {
      fail("Choose a clip to stream first.");
      return;
    }
    if (sessionMode === "mic") {
      if (!window.isSecureContext) {
        fail("Microphone capture needs a secure context (https or localhost).");
        return;
      }
      if (!navigator.mediaDevices?.getUserMedia) {
        fail("This browser does not support microphone capture.");
        return;
      }
    }

    let ws: WebSocket;
    try {
      ws = new WebSocket(analyzeStreamUrl());
    } catch {
      fail(`Could not open a WebSocket to ${wsBase()}.`);
      return;
    }
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onerror = () => {
      fail(`Lost the streaming connection to ${wsBase()}. Check that the backend is running.`);
    };

    ws.onclose = () => {
      const s = stateRef.current;
      if (s === "live" || s === "connecting") {
        fail("The server closed the stream unexpectedly.");
      } else if (s === "stopping") {
        fail("The stream closed before a final report arrived.");
      }
    };

    ws.onmessage = (ev) => {
      let data: StreamServerEvent;
      try {
        data = JSON.parse(ev.data);
      } catch {
        return; // ignore malformed frames
      }
      if (data.type === "analysis.window") {
        setEvents((prev) => [...prev, data]);
      } else if (data.type === "analysis.final") {
        setFinalReport(data.report);
        // Only a microphone session is "your voice" — the human baseline Red Team compares against.
        if (sessionModeRef.current === "mic") onReportReady?.(data.report);
        setSession("idle");
        cleanup();
      } else if (data.type === "error") {
        fail(data.detail);
      }
    };

    ws.onopen = async () => {
      try {
        const tap = {
          onChunk: (chunk: Float32Array) => {
            if (ws.readyState === WebSocket.OPEN) ws.send(chunk.buffer);
          },
          onLevel: (dbfs: number) => setLevelDbfs(dbfs),
        };
        const mic =
          sessionMode === "mic"
            ? await startMicCapture(tap)
            : await startClipCapture(sessionClip!.blob, {
                ...tap,
                onProgress: (t, d) => setClipProgress({ t, d }),
                onEnded: () => stopRef.current(),
              });
        if (wsRef.current !== ws) {
          mic.stop(); // session was cancelled while the permission prompt was open
          return;
        }
        micRef.current = mic;
        ws.send(
          JSON.stringify({
            type: "start",
            sample_rate: mic.sampleRate,
            encoding: "f32le",
            source: sessionMode === "mic" ? "mic" : "file",
          })
        );
        setSession("live");
      } catch (err) {
        fail(sessionMode === "mic" ? micErrorMessage(err) : err instanceof Error ? err.message : String(err));
      }
    };
  }, [cleanup, fail, onReportReady, mode, clip]);

  const stop = useCallback(() => {
    try {
      micRef.current?.stop();
    } catch {}
    micRef.current = null;
    setLevelDbfs(null);
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      setSession("stopping");
      ws.send(JSON.stringify({ type: "stop" }));
      finalizeTimer.current = setTimeout(
        () => fail("Timed out waiting for the final report."),
        FINALIZE_TIMEOUT_MS
      );
    } else {
      setSession("idle");
      cleanup();
    }
  }, [cleanup, fail]);
  stopRef.current = stop;

  const latest = events[events.length - 1];
  const busy = state === "connecting" || state === "live" || state === "stopping";

  return (
    <div className="stack">
      <div className="page-intro">
        <h2>Live Trust</h2>
        <p>
          Every half second the detector scores the newest audio and updates a rolling probability, so you can
          watch how fast the evidence becomes conclusive. Speak into your microphone, or stream a clip to see
          an AI voice analyzed in real time.
        </p>
      </div>

      <section className="panel" aria-label="Live session controls">
        <div className="mode-switch" role="radiogroup" aria-label="Input">
          {(
            [
              ["mic", "Microphone", "mic"],
              ["clip", "Stream a clip", "file"],
            ] as const
          ).map(([m, label, icon]) => (
            <button
              key={m}
              type="button"
              role="radio"
              aria-checked={mode === m}
              className="mode-option"
              disabled={busy}
              onClick={() => setMode(m)}
            >
              <Icon name={icon} size={14} />
              {label}
            </button>
          ))}
        </div>

        {mode === "mic" ? (
          <p className="panel-sub" style={{ margin: "12px 0 16px", maxWidth: "70ch" }}>
            <strong style={{ color: "var(--text-dim)" }}>Testing an AI voice?</strong> Don&rsquo;t play it from a
            phone into the microphone. The speaker, room echo and mic reshape the audio into a channel this model
            wasn&rsquo;t trained on (a <em>replay</em> attack), and results become unreliable in both directions.
            Use{" "}
            <button type="button" className="link-btn" onClick={() => setMode("clip")} disabled={busy}>
              Stream a clip
            </button>{" "}
            instead: it plays through your speakers and sends the same samples to the detector digitally.
          </p>
        ) : (
          <div className="clip-picker">
            <div className="clip-picker-row" role="list" aria-label="Clips to stream">
              {samples.map((smp) => (
                <button
                  type="button"
                  role="listitem"
                  key={smp.id}
                  className={`clip-option ${clip?.sampleId === smp.id ? "active" : ""}`}
                  onClick={() => pickSample(smp)}
                  disabled={busy}
                  aria-pressed={clip?.sampleId === smp.id}
                >
                  <span
                    className="dot"
                    aria-hidden
                    style={{ background: smp.truth === "bonafide" ? "var(--green)" : "var(--red)" }}
                  />
                  {smp.title}
                </button>
              ))}
              <button
                type="button"
                role="listitem"
                className={`clip-option ${clip && !clip.sampleId ? "active" : ""}`}
                onClick={() => clipInputRef.current?.click()}
                disabled={busy}
              >
                <Icon name="upload" size={13} />
                {clip && !clip.sampleId ? clip.name : "Your own file…"}
              </button>
            </div>
            <input
              ref={clipInputRef}
              type="file"
              accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) setClip({ name: f.name, blob: f });
                e.target.value = "";
              }}
            />
          </div>
        )}

        <div className="control-bar">
          <div className="control-group">
            {!busy ? (
              <button className="btn btn-primary btn-lg" onClick={start} disabled={mode === "clip" && !clip}>
                <Icon name={mode === "mic" ? "mic" : "play"} size={18} />
                {mode === "mic"
                  ? finalReport || events.length
                    ? "Start new session"
                    : "Start listening"
                  : clip
                  ? `Stream “${clip.name}”`
                  : "Choose a clip above"}
              </button>
            ) : (
              <button
                className="btn btn-danger btn-lg"
                onClick={state === "connecting" ? () => (setSession("idle"), cleanup()) : stop}
                disabled={state === "stopping"}
              >
                {state === "stopping" ? <span className="spinner" aria-hidden /> : <Icon name="stop" size={16} />}
                {state === "stopping" ? "Finalizing…" : state === "connecting" ? "Cancel" : "Stop & finalize"}
              </button>
            )}
            <span className={`live-indicator ${state === "live" ? "on" : ""}`} role="status" aria-live="polite">
              <span className="dot" aria-hidden />
              {state === "connecting" &&
                (mode === "mic" ? "Connecting — allow microphone access if prompted" : "Connecting…")}
              {state === "live" && (
                <>
                  {sessionModeRef.current === "clip" ? "Streaming" : "Live"}{" "}
                  <span className="num">
                    {sessionModeRef.current === "clip" && clipProgress
                      ? `${clock(clipProgress.t * 1000)} / ${clock(clipProgress.d * 1000)}`
                      : clock(elapsed)}
                  </span>
                </>
              )}
              {state === "stopping" && "Building final report…"}
              {state === "idle" && (finalReport ? "Session complete" : "Idle")}
              {state === "error" && "Stopped"}
            </span>
          </div>
          {latest && <StatusChip status={latest.status} />}
        </div>

        {state === "live" && (
          <div style={{ marginTop: 16 }}>
            <LevelMeter dbfs={levelDbfs} />
          </div>
        )}
      </section>

      {error && (
        <Callout tone="error" title="Live session stopped">
          {error}
        </Callout>
      )}

      {(busy || (events.length > 0 && !finalReport)) && (
        <section className="panel" aria-labelledby="rolling-title">
          <div className="panel-head">
            <h3 id="rolling-title">Rolling probability</h3>
            <span className="panel-sub">{events.length} windows</span>
          </div>
          <RollingProbabilityChart events={events} />
          <div className="stat-grid">
            <div className="stat">
              <div className="stat-label">Current reading</div>
              <div className="stat-value lg">
                {latest ? (
                  <>
                    {leaning(latest.rolling_probability)!.text}{" "}
                    <span className="stat-unit">{leaning(latest.rolling_probability)!.side}</span>
                  </>
                ) : (
                  "—"
                )}
              </div>
            </div>
            <div className="stat">
              <div className="stat-label">Analysis confidence</div>
              <div className="stat-value lg">{latest ? pct(latest.analysis_confidence) : "—"}</div>
            </div>
            <div className="stat">
              <div className="stat-label">Time to confidence</div>
              <div className="stat-value lg">{ms(latest?.time_to_confidence_ms)}</div>
            </div>
            <div className="stat">
              <div className="stat-label">Speech in window</div>
              <div className="stat-value lg">{latest ? pct(latest.speech_ratio) : "—"}</div>
            </div>
          </div>
        </section>
      )}

      {finalReport && (
        <div className="stack">
          <div className="panel-head" style={{ marginBottom: 0 }}>
            <h3 className="section-title" style={{ fontSize: "1rem" }}>
              Final report
            </h3>
            <span className="panel-sub">
              {sessionModeRef.current === "mic"
                ? "Used as the human baseline in Red Team"
                : "Streamed clip (not used as your human baseline)"}
            </span>
          </div>
          <ReportView report={finalReport} />
        </div>
      )}
    </div>
  );
}
