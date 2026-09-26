// Types mirroring api/CONTRACT.md (v1). Keep in lockstep with that file.
// All probabilities are p_synthetic in [0,1] (1.0 = synthetic).
// No field is ever fabricated on the frontend: missing/uncomputed values stay
// null/undefined and the UI must say so explicitly.

export type Status =
  | "insufficient_evidence"
  | "likely_human"
  | "inconclusive"
  | "likely_synthetic";

export interface FileMeta {
  name: string | null;
  container: string | null;
  codec: string | null;
  sample_rate: number | null;
  channels: number | null;
  bit_depth: number | null;
  duration_s: number | null;
  bitrate: number | null;
  metadata: Record<string, unknown> | null;
}

export interface DetectorInfo {
  name: string;
  version: string;
  raw_score: number | null;
  latency_ms: number | null;
}

export interface TimelineWindow {
  start_ms: number;
  end_ms: number;
  synthetic_probability: number;
  analysis_confidence: number | null;
  detector_scores?: Record<string, number> | null;
  evidence?: unknown[] | null;
}

export interface SuspiciousRegion {
  start_ms: number;
  end_ms: number;
  peak_probability: number;
  reason: string;
}

export interface EvidenceEntry {
  summary: string;
  suspicion: number | null; // 0..1 or null if not a scoring branch
  features: Record<string, unknown>;
  flags: string[];
  used_in_score: boolean;
}

export interface BranchLogEntry {
  technique: string;
  ran: boolean;
  reason: string;
  runtime_ms: number;
}

export interface AnalysisReport {
  id: string;
  file: FileMeta;
  synthetic_probability: number;
  cm_score: number;
  analysis_confidence: number;
  status: Status;
  decision_threshold: number;
  detector: DetectorInfo;
  timeline: TimelineWindow[];
  suspicious_regions: SuspiciousRegion[];
  time_to_confidence_ms: number | null;
  techniques_run: string[];
  why_run: Record<string, string>;
  evidence: Record<string, EvidenceEntry>;
  branch_log: BranchLogEntry[];
  processing_ms: number;
  disclaimer: string;
}

// ---- /health ----
export interface HealthResponse {
  status: string;
  version: string;
  detector: { name: string; version: string; device: string };
  grok_available: boolean;
}

// ---- WS /analyze/stream ----
export interface StreamStartMessage {
  type: "start";
  sample_rate: number;
  encoding: "f32le" | "s16le";
  source: "mic" | "grok" | "file";
}

export interface StreamStopMessage {
  type: "stop";
}

export interface ReadyEvent {
  type: "ready";
}

export interface AnalysisWindowEvent {
  type: "analysis.window";
  t_ms: number;
  start_ms: number;
  end_ms: number;
  synthetic_probability: number;
  rolling_probability: number;
  analysis_confidence: number;
  status: Status;
  time_to_confidence_ms: number | null;
  level_dbfs: number | null;
  speech_ratio: number | null;
}

export interface AnalysisFinalEvent {
  type: "analysis.final";
  report: AnalysisReport;
}

export interface ErrorEvent {
  type: "error";
  detail: string;
}

export type StreamServerEvent =
  | ReadyEvent
  | AnalysisWindowEvent
  | AnalysisFinalEvent
  | ErrorEvent;

// ---- /redteam ----
export type RedteamSource = "grok_voice" | "grok_tts" | "cached_fixture";

export interface RedteamGrokRequest {
  text?: string;
  voice?: string;
}

export interface RedteamGrokResponse {
  source: RedteamSource;
  text: string;
  audio_wav_b64: string;
  sample_rate: number;
  report: AnalysisReport;
  note?: string;
}

export interface RedteamStreamStart {
  type: "start";
  text?: string;
}

export interface RedteamAudioChunkEvent {
  type: "audio.chunk";
  pcm16_b64: string;
  sample_rate: number;
  seq: number;
}

export interface RedteamDoneEvent {
  type: "redteam.done";
  source: RedteamSource;
  report: AnalysisReport;
}

export type RedteamStreamServerEvent =
  | RedteamAudioChunkEvent
  | AnalysisWindowEvent
  | RedteamDoneEvent
  | ErrorEvent;

// ---- /eval/summary ----
// Shape is intentionally open: "Returns outputs/eval_summary.json verbatim."
// The UI renders whatever is present generically and never fabricates data.
export type EvalHistogram = {
  bins: number[];
  bonafide: number[];
  spoof: number[];
};

export type EvalSummary = Record<string, unknown>;

// ---- /explain ----
export interface ExplainRequest {
  report: AnalysisReport;
}

export interface ExplainResponse {
  text: string;
  model: string;
  grounded: boolean;
}

export interface ApiErrorBody {
  detail: string;
}
