"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { AnalysisReport } from "@/lib/types";
import { analyzeFile, ApiError, describeError, isAbort } from "@/lib/api";
import { decodeAudioFileForWaveform } from "@/lib/audio";
import { bytes } from "@/lib/format";
import { fetchSampleFile, fetchSampleReport, loadSampleManifest, truthLabel, type Sample } from "@/lib/samples";
import ReportView from "./ReportView";
import SampleLibrary from "./SampleLibrary";
import Callout from "./Callout";
import Icon from "./Icon";

const ACCEPT = "audio/*,video/mp4,video/webm,.wav,.mp3,.m4a,.ogg,.oga,.opus,.flac,.aac,.webm,.mp4";

function useElapsed(running: boolean) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!running) return;
    const t0 = performance.now();
    setElapsed(0);
    const id = setInterval(() => setElapsed(performance.now() - t0), 100);
    return () => clearInterval(id);
  }, [running]);
  return elapsed;
}

export default function ForensicLab({
  onReportReady,
}: {
  onReportReady?: (report: AnalysisReport) => void;
}) {
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [waveformPeaks, setWaveformPeaks] = useState<number[] | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [sample, setSample] = useState<Sample | null>(null);
  const [precomputedNote, setPrecomputedNote] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const ctrlRef = useRef<AbortController | null>(null);
  const dragDepth = useRef(0);
  const resultRef = useRef<HTMLDivElement>(null);
  const elapsed = useElapsed(loading);

  useEffect(() => () => ctrlRef.current?.abort(), []);

  // Object URL for listen-along playback; revoked when replaced or on unmount.
  useEffect(() => {
    if (!file || file.size === 0) {
      setAudioUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const showResult = () =>
    requestAnimationFrame(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));

  const handleFile = useCallback(
    async (f: File, fromSample: Sample | null = null) => {
      // A newer upload supersedes any in-flight one; its late response is ignored.
      ctrlRef.current?.abort();
      const ctrl = new AbortController();
      ctrlRef.current = ctrl;

      setError(null);
      setReport(null);
      setFile(f);
      setSample(fromSample);
      setPrecomputedNote(null);
      setWaveformPeaks(null);

      if (f.size === 0) {
        setError("This file is empty.");
        return;
      }

      setLoading(true);
      decodeAudioFileForWaveform(f)
        .then((res) => {
          if (!ctrl.signal.aborted && res) setWaveformPeaks(res.peaks);
        })
        .catch(() => {});

      try {
        const result = await analyzeFile(f, ctrl.signal);
        if (ctrl.signal.aborted) return;
        setReport(result);
        onReportReady?.(result);
        showResult();
      } catch (err) {
        if (isAbort(err) || ctrl.signal.aborted) return;
        // Samples stay demoable without a backend: fall back to the reference report the real API produced
        // when the library was built, and say so. Uploads never fall back.
        if (fromSample && !(err instanceof ApiError)) {
          try {
            const [ref, manifest] = await Promise.all([fetchSampleReport(fromSample), loadSampleManifest()]);
            if (ctrl.signal.aborted) return;
            const when = manifest?.generated_at ? new Date(manifest.generated_at).toLocaleDateString() : "build time";
            setReport(ref);
            setPrecomputedNote(
              `The ProofVoice API is not reachable right now, so this is the reference result ${
                ref.detector?.name ?? "the detector"
              } produced for this exact clip on ${when}. Uploads and live analysis need the backend.`
            );
            showResult();
            return;
          } catch {
            /* fall through to the original error */
          }
        }
        setError(describeError(err));
      } finally {
        if (ctrlRef.current === ctrl) setLoading(false);
      }
    },
    [onReportReady]
  );

  const analyzeSample = useCallback(
    async (s: Sample) => {
      try {
        const f = await fetchSampleFile(s);
        await handleFile(f, s);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Could not load the sample.");
      }
    },
    [handleFile]
  );

  const cancel = () => {
    ctrlRef.current?.abort();
    setLoading(false);
  };

  const openPicker = () => inputRef.current?.click();

  const onDrop = (e: React.DragEvent<HTMLElement>) => {
    e.preventDefault();
    dragDepth.current = 0;
    setDragActive(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  };

  const dropHandlers = {
    onDragEnter: (e: React.DragEvent) => {
      e.preventDefault();
      dragDepth.current += 1;
      setDragActive(true);
    },
    onDragOver: (e: React.DragEvent) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
    },
    // Counting enter/leave avoids flicker when the pointer crosses child elements.
    onDragLeave: () => {
      dragDepth.current = Math.max(0, dragDepth.current - 1);
      if (dragDepth.current === 0) setDragActive(false);
    },
    onDrop,
  };

  const compact = !!file && (loading || !!report);

  return (
    <div className="stack">
      {!compact && (
        <div className="page-intro">
          <h2>Forensic Lab</h2>
          <p>
            Upload a recording, or pick a sample below, to get a synthetic-speech probability, a per-window
            trust timeline, suspicious regions and the evidence behind them. Audio is analyzed by the ProofVoice
            backend and is not stored.
          </p>
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) handleFile(f);
          e.target.value = ""; // allow re-selecting the same file
        }}
      />

      {compact ? (
        <div className={`dropzone compact ${dragActive ? "active" : ""}`} {...dropHandlers}>
          <div className="file-chip">
            <span className="dropzone-icon" style={{ width: 40, height: 40, borderRadius: 12 }}>
              <Icon name="file" size={20} />
            </span>
            <div style={{ minWidth: 0 }}>
              <div className="name">{sample ? sample.title : file!.name}</div>
              <div className="meta">
                {sample ? `Sample · ${sample.group} · ` : ""}
                {bytes(file!.size)}
                {!sample && file!.type ? ` · ${file!.type}` : ""}
              </div>
            </div>
          </div>
          <div className="control-group">
            {loading ? (
              <button className="btn btn-sm" onClick={cancel}>
                Cancel
              </button>
            ) : (
              <button className="btn btn-sm" onClick={openPicker}>
                <Icon name="upload" size={14} />
                <span className="vh-sm">Analyze another file</span>
                <span aria-hidden className="show-sm">New file</span>
              </button>
            )}
          </div>
        </div>
      ) : (
        <button
          type="button"
          className={`dropzone ${dragActive ? "active" : ""}`}
          onClick={openPicker}
          aria-describedby="dropzone-hint"
          {...dropHandlers}
        >
          <span className="dropzone-icon" aria-hidden>
            <Icon name="upload" size={24} />
          </span>
          <span className="dropzone-title">
            {dragActive ? "Release to analyze" : "Drop an audio file, or click to browse"}
          </span>
          <span className="dropzone-hint" id="dropzone-hint">
            WAV, MP3, M4A, OGG, FLAC, MP4 or any ffmpeg-decodable format
          </span>
        </button>
      )}

      <SampleLibrary
        variant={compact ? "strip" : "grid"}
        onAnalyze={analyzeSample}
        activeId={sample?.id ?? null}
        busy={loading}
      />

      {loading && (
        <div className="panel appear" role="status" aria-live="polite">
          <div className="progress-card">
            <span className="spinner" aria-hidden />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8125rem" }}>
                <span>Running detector and forensic branches…</span>
                <span className="num" style={{ color: "var(--text-faint)" }}>
                  {(elapsed / 1000).toFixed(1)}s
                </span>
              </div>
              <div className="progress-track" style={{ marginTop: 10 }} />
            </div>
          </div>
        </div>
      )}

      {error && (
        <Callout
          tone="error"
          title="Analysis failed"
          action={
            file && file.size > 0 ? (
              <button className="btn btn-sm" onClick={() => handleFile(file, sample)}>
                <Icon name="refresh" size={14} />
                Retry
              </button>
            ) : undefined
          }
        >
          {error}
        </Callout>
      )}

      {report && !loading && (
        <div ref={resultRef} style={{ scrollMarginTop: 140 }}>
          <ReportView
            report={report}
            waveformPeaks={waveformPeaks}
            audioUrl={audioUrl}
            precomputedNote={precomputedNote}
            groundTruth={sample ? { truth: sample.truth, label: truthLabel(sample.truth) } : null}
          />
        </div>
      )}
    </div>
  );
}
