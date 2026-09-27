"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  AnalysisReport,
  AnalysisWindowEvent,
  RedteamSource,
  RedteamStreamServerEvent,
} from "@/lib/types";
import { redteamGrok, redteamStreamUrl, describeError, isAbort, wsBase } from "@/lib/api";
import { base64ToInt16Array, PcmStreamPlayer } from "@/lib/audio";
import { pct } from "@/lib/format";
import { useHealth } from "@/lib/useHealth";
import RollingProbabilityChart from "./RollingProbabilityChart";
import ReportView from "./ReportView";
import StatusChip from "./StatusChip";
import { LeaningValue } from "./EvidenceScale";
import Callout from "./Callout";
import Icon from "./Icon";

const SOURCE_LABEL: Record<RedteamSource, string> = {
  grok_voice: "Grok Voice (realtime)",
  grok_tts: "Grok streaming TTS (fallback)",
  cached_fixture: "Cached Grok clip (offline fallback)",
};

const MAX_TEXT = 280;

function CompareCard({ title, report }: { title: string; report: AnalysisReport }) {
  return (
    <div className="panel compare-card">
      <span className="eyebrow">{title}</span>
      <LeaningValue probability={report.synthetic_probability} status={report.status} size="md" />
      <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
        <StatusChip status={report.status} />
        <span className="panel-sub">confidence {pct(report.analysis_confidence)}</span>
      </div>
    </div>
  );
}

export default function RedTeam({
  lastHumanReport,
  onGoLive,
}: {
  lastHumanReport?: AnalysisReport | null;
  onGoLive?: () => void;
}) {
  const health = useHealth();
  const [text, setText] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [events, setEvents] = useState<AnalysisWindowEvent[]>([]);
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [source, setSource] = useState<RedteamSource | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [wavUrl, setWavUrl] = useState<string | null>(null);
  const [mode, setMode] = useState<"stream" | "fallback" | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const playerRef = useRef<PcmStreamPlayer | null>(null);
  const fetchCtrl = useRef<AbortController | null>(null);

  const closeSocket = () => {
    const ws = wsRef.current;
    wsRef.current = null;
    if (ws) {
      ws.onclose = ws.onerror = ws.onmessage = ws.onopen = null;
      try {
        ws.close();
      } catch {}
    }
  };

  const reset = () => {
    closeSocket();
    fetchCtrl.current?.abort();
    // Dispose the previous run's player (not on ws close: audio arrives faster than real time and may still be playing).
    playerRef.current?.close();
    playerRef.current = null;
    setEvents([]);
    setReport(null);
    setSource(null);
    setNote(null);
    setWavUrl(null);
    setError(null);
  };

  useEffect(
    () => () => {
      closeSocket();
      fetchCtrl.current?.abort();
      playerRef.current?.close();
    },
    []
  );

  const cancel = () => {
    closeSocket();
    fetchCtrl.current?.abort();
    playerRef.current?.close();
    playerRef.current = null;
    setRunning(false);
  };

  const runStream = useCallback(() => {
    reset();
    setMode("stream");
    setRunning(true);

    let ws: WebSocket;
    try {
      ws = new WebSocket(redteamStreamUrl());
    } catch {
      setError(`Could not open a WebSocket to ${wsBase()}.`);
      setRunning(false);
      return;
    }
    wsRef.current = ws;
    let done = false;

    ws.onerror = () => {
      setError(`Red-team stream error. Check that the backend is running at ${wsBase()}.`);
      setRunning(false);
    };

    ws.onopen = () => {
      ws.send(JSON.stringify({ type: "start", text: text.trim() || undefined }));
    };

    ws.onmessage = (ev) => {
      let data: RedteamStreamServerEvent;
      try {
        data = JSON.parse(ev.data);
      } catch {
        return; // ignore malformed frames
      }
      if (data.type === "audio.chunk") {
        if (!playerRef.current) playerRef.current = new PcmStreamPlayer(data.sample_rate);
        playerRef.current.enqueue(base64ToInt16Array(data.pcm16_b64));
      } else if (data.type === "analysis.window") {
        setEvents((prev) => [...prev, data]);
      } else if (data.type === "redteam.done") {
        done = true;
        setReport(data.report);
        setSource(data.source);
        setRunning(false);
        closeSocket();
      } else if (data.type === "error") {
        setError(data.detail);
        setRunning(false);
      }
    };

    ws.onclose = () => {
      if (!done) setError((e) => e ?? "The red-team stream closed before a report arrived.");
      setRunning(false);
    };
  }, [text]);

  const runFallback = useCallback(async () => {
    reset();
    setMode("fallback");
    setRunning(true);
    const ctrl = new AbortController();
    fetchCtrl.current = ctrl;
    try {
      const result = await redteamGrok({ text: text.trim() || undefined }, ctrl.signal);
      if (ctrl.signal.aborted) return;
      setReport(result.report);
      setSource(result.source);
      setNote(result.note ?? null);
      // Only offer playback when the payload is a real RIFF/WAV container ("RIFF" → "UklGR" in base64).
      if (result.audio_wav_b64?.startsWith("UklGR")) {
        setWavUrl(`data:audio/wav;base64,${result.audio_wav_b64}`);
      }
    } catch (err) {
      if (isAbort(err) || ctrl.signal.aborted) return;
      setError(describeError(err));
    } finally {
      if (fetchCtrl.current === ctrl) setRunning(false);
    }
  }, [text]);

  const latest = events[events.length - 1];
  const grokLive = health.kind === "ok" ? health.data.grok_available : null;

  return (
    <div className="stack">
      <div className="page-intro">
        <h2>Red Team</h2>
        <p>
          Grok Voice speaks a line, and the exact same audio bytes stream through the same detector in real time.
          The result is compared against your last Live Trust recording.
        </p>
      </div>

      <section className="panel" aria-label="Generate adversarial sample">
        <form
          className="field-row"
          onSubmit={(e) => {
            e.preventDefault();
            if (!running) runStream();
          }}
        >
          <label htmlFor="redteam-text" className="sr-only">
            Line for Grok to speak
          </label>
          <input
            id="redteam-text"
            className="text-input"
            type="text"
            placeholder="Optional: a line for Grok to speak…"
            value={text}
            maxLength={MAX_TEXT}
            onChange={(e) => setText(e.target.value)}
            disabled={running}
            autoComplete="off"
          />
          {running ? (
            <button type="button" className="btn btn-danger" onClick={cancel}>
              <Icon name="stop" size={14} />
              Cancel
            </button>
          ) : (
            <button type="submit" className="btn btn-primary">
              <Icon name="spark" size={15} />
              Generate &amp; analyze
            </button>
          )}
        </form>
        <div className="control-bar" style={{ marginTop: 10 }}>
          <span className="field-hint" style={{ marginTop: 0 }}>
            {grokLive === false
              ? "No xAI key on the server: a cached, previously generated Grok clip will be used and labeled as such."
              : "Audio plays as it streams in. Leave blank for a default line."}
          </span>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={runFallback}
            disabled={running}
            title="POST /redteam/grok — generate first, then analyze the whole clip"
          >
            Non-streaming fallback
          </button>
        </div>

        {source && (
          <div className="source-line appear">
            <span>Audio source</span>
            <span className={`badge ${source === "cached_fixture" ? "badge-warn" : "badge-used"}`}>
              {SOURCE_LABEL[source]}
            </span>
            {note && <span className="panel-sub">{note}</span>}
          </div>
        )}
        {wavUrl && (
          <audio controls src={wavUrl} style={{ width: "100%", marginTop: 12 }}>
            Your browser cannot play this clip.
          </audio>
        )}
      </section>

      {error && (
        <Callout tone="error" title="Red-team run failed">
          {error}
        </Callout>
      )}

      {mode === "stream" && (running || events.length > 0) && !report && (
        <section className="panel" aria-labelledby="rt-live-title">
          <div className="panel-head">
            <h3 id="rt-live-title">Live analysis of generated audio</h3>
            {latest && <StatusChip status={latest.status} />}
          </div>
          <RollingProbabilityChart events={events} />
        </section>
      )}

      {mode === "fallback" && running && (
        <div className="panel appear" role="status">
          <div className="progress-card">
            <span className="spinner" aria-hidden />
            <span style={{ fontSize: "0.8125rem" }}>Generating speech with Grok, then analyzing…</span>
          </div>
        </div>
      )}

      {report && (
        <div className="stack">
          {lastHumanReport ? (
            <div className="grid-2 appear">
              <CompareCard title="Your voice (Live Trust)" report={lastHumanReport} />
              <CompareCard title={`Grok · ${source ? SOURCE_LABEL[source] : "sample"}`} report={report} />
            </div>
          ) : (
            <Callout
              tone="info"
              title="No human baseline yet"
              action={
                onGoLive && (
                  <button className="btn btn-sm" onClick={onGoLive}>
                    <Icon name="mic" size={14} />
                    Go to Live Trust
                  </button>
                )
              }
            >
              Record a Live Trust session to see your voice and Grok side by side.
            </Callout>
          )}
          <ReportView report={report} />
        </div>
      )}
    </div>
  );
}
