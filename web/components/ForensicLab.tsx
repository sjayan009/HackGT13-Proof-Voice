"use client";

import { useCallback, useRef, useState } from "react";
import type { AnalysisReport } from "@/lib/types";
import { analyzeFile, ApiError } from "@/lib/api";
import { decodeAudioFileForWaveform } from "@/lib/audio";
import ReportView from "./ReportView";

export default function ForensicLab({
  onReportReady,
}: {
  onReportReady?: (report: AnalysisReport) => void;
}) {
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [waveformPeaks, setWaveformPeaks] = useState<number[] | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (file: File) => {
      setError(null);
      setLoading(true);
      setReport(null);
      setFileName(file.name);
      setWaveformPeaks(null);

      decodeAudioFileForWaveform(file).then((res) => {
        if (res) setWaveformPeaks(res.peaks);
      });

      try {
        const result = await analyzeFile(file);
        setReport(result);
        onReportReady?.(result);
      } catch (err) {
        if (err instanceof ApiError) {
          setError(err.message);
        } else if (err instanceof TypeError) {
          setError(
            "Could not reach the ProofVoice API. Is the backend running at the configured NEXT_PUBLIC_API_BASE_URL?"
          );
        } else {
          setError(err instanceof Error ? err.message : "Unknown error");
        }
      } finally {
        setLoading(false);
      }
    },
    [onReportReady]
  );

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragActive(false);
      const file = e.dataTransfer.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div>
      <div
        className={`dropzone ${dragActive ? "active" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="audio/*,video/mp4,.wav,.mp3,.m4a,.ogg,.mp4"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
        />
        <div style={{ fontSize: 15, marginBottom: 6 }}>
          {fileName ? `Selected: ${fileName}` : "Drop an audio file here, or click to browse"}
        </div>
        <div style={{ fontSize: 12 }}>
          WAV, MP3, M4A, OGG, MP4 audio, or any ffmpeg-decodable format
        </div>
      </div>

      {loading && (
        <div className="panel" style={{ marginTop: 16 }}>
          Analyzing…
        </div>
      )}

      {error && (
        <div className="error-state" style={{ marginTop: 16 }}>
          {error}
        </div>
      )}

      {report && !loading && (
        <div style={{ marginTop: 16 }}>
          <ReportView report={report} waveformPeaks={waveformPeaks} />
        </div>
      )}
    </div>
  );
}
