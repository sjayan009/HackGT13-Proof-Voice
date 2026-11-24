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

// =====================================================================
// v2 — Guardian (live telephone protection) and Aegis engine
// =====================================================================

export type RiskState = "no_suspicious_evidence" | "inconclusive" | "suspicious" | "analysis_unavailable";
export type Consent = "granted" | "declined" | "simulated";
export type Scenario = "genuine" | "synthetic" | "mixed";
export type ComputeAction = "infer" | "skip_silence" | "skip_clear" | "escalate" | "shed";
export type EnginePolicy = "fifo" | "dynamic" | "edf" | "adaptive";
export type VerificationOutcome = "verified_genuine" | "could_not_reach" | "confirmed_scam" | "skipped";

// ---- WS /guardian/live (server → client only) ----
export interface GuardianHelloEvent {
  type: "hello";
  user_id: string;
  dev_mode: boolean;
}

export interface CallStartedEvent {
  type: "call.started";
  session_id: string;
  call_sid: string;
  caller: string | null;
  consent: Consent;
  started_at: string;
  scenario: Scenario | null;
}

export interface GuardianWindow {
  start_ms: number;
  end_ms: number;
  synthetic_probability: number;
  usable: boolean;
  weight: number;
  speech_s: number;
  snr_db: number | null;
  level_dbfs: number | null;
  packet_loss: number | null;
}

export interface CallUpdateEvent {
  type: "call.update";
  session_id: string;
  t_ms: number;
  state: RiskState;
  evidence: number; // S/H in [0, 1.5]; alert threshold at 1.0
  reason: string;
  window: GuardianWindow | null; // null = skipped hop
  compute: { policy: string; action: ComputeAction; latency_ms: number | null };
}

export interface CallAlertEvent {
  type: "call.alert";
  session_id: string;
  t_ms: number;
  evidence: number;
  message: string;
  sms_sent: boolean;
}

export interface CallEndedEvent {
  type: "call.ended";
  session_id: string;
  final_state: RiskState;
  peak_evidence: number;
  duration_ms: number;
  usable_speech_s: number;
  windows_inferred: number;
  windows_skipped: number;
}

export type GuardianLiveEvent =
  | GuardianHelloEvent
  | CallStartedEvent
  | CallUpdateEvent
  | CallAlertEvent
  | CallEndedEvent;

// ---- Guardian REST ----
// All simulated audio goes through the 8 kHz mu-law phone path (no channel choice).
export interface SimulateRequest {
  scenario: Scenario;
}
export interface SimulateResponse {
  session_id: string;
}
export interface VerificationRequest {
  contact_id: string | null;
  outcome: VerificationOutcome;
}

// ---- Supabase tables (infrastructure/supabase/migrations/0001_guardian_schema.sql) ----
export interface GuardianLine {
  id: string;
  user_id: string;
  twilio_number: string;
  recipient_phone: string;
  display_name: string | null;
  created_at: string;
}

export interface TrustedContact {
  id: string;
  user_id: string;
  name: string;
  relationship: string | null;
  phone: string;
  created_at: string;
}

export interface CallSessionRow {
  id: string;
  user_id: string;
  line_id: string | null;
  call_sid: string;
  caller_number: string | null;
  consent: Consent;
  started_at: string;
  ended_at: string | null;
  final_state: RiskState | null;
  peak_evidence: number | null;
  alerted: boolean;
}

export interface RiskEventRow {
  id: number;
  session_id: string;
  user_id: string;
  t_ms: number | null;
  kind: "state" | "alert" | "verification" | "unavailable";
  state: string | null;
  evidence: number | null;
  detail: Record<string, unknown>;
  created_at: string;
}

// ---- Engine / Inference Control Center ----
export interface Percentiles {
  p50: number | null;
  p95: number | null;
  p99?: number | null;
  n?: number;
}

export interface EngineMetrics {
  policy: EnginePolicy | string;
  backend: string;
  device: string | null;
  active_calls: number;
  queue_depth: number;
  window_s: number;
  hop_s: number;
  deadline_ms: number;
  latency_ms: Percentiles;
  compute_ms: Percentiles;
  queue_wait_ms: Percentiles;
  windows_per_s: { demand: number; inferred: number };
  actions: Partial<Record<ComputeAction, number>>;
  batch_size: { mean: number | null; max: number | null; hist: Record<string, number> };
  gpu: null | {
    util_pct: number | null;
    mem_used_mb: number | null;
    mem_total_mb: number | null;
    power_w: number | null;
    temp_c: number | null;
  };
  t: number; // unix seconds
}

export interface LoadTestRequest {
  calls: number;
  seconds: number;
  scenario: Scenario;
}

export interface BenchmarkReport {
  name: string;
  created: string;
  data: unknown; // verbatim JSON from outputs/benchmarks/*.json
}
