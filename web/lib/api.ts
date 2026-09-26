// Thin API client for the ProofVoice backend. Mirrors api/CONTRACT.md exactly.
// Never invents fields or endpoints not in the contract.

import type {
  AnalysisReport,
  ApiErrorBody,
  EvalSummary,
  ExplainRequest,
  ExplainResponse,
  HealthResponse,
  RedteamGrokRequest,
  RedteamGrokResponse,
} from "./types";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";
export const WS_BASE_URL =
  process.env.NEXT_PUBLIC_WS_BASE_URL || "ws://127.0.0.1:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.name = "ApiError";
  }
}

async function parseJsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `Request failed with status ${res.status}`;
    try {
      const body = (await res.json()) as ApiErrorBody;
      if (body?.detail) detail = body.detail;
    } catch {
      // ignore parse failure, keep generic detail
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/health`, { cache: "no-store" });
  return parseJsonOrThrow<HealthResponse>(res);
}

export async function analyzeFile(file: File): Promise<AnalysisReport> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE_URL}/analyze/file`, {
    method: "POST",
    body: form,
  });
  return parseJsonOrThrow<AnalysisReport>(res);
}

export async function redteamGrok(
  req: RedteamGrokRequest
): Promise<RedteamGrokResponse> {
  const res = await fetch(`${API_BASE_URL}/redteam/grok`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return parseJsonOrThrow<RedteamGrokResponse>(res);
}

export async function explainReport(
  report: AnalysisReport
): Promise<ExplainResponse> {
  const body: ExplainRequest = { report };
  const res = await fetch(`${API_BASE_URL}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return parseJsonOrThrow<ExplainResponse>(res);
}

// GET /eval/summary — 404 means "not computed yet", which is a valid, expected
// state (never rendered as an error, never backfilled with placeholder data).
export type EvalSummaryResult =
  | { kind: "ok"; data: EvalSummary }
  | { kind: "not_computed" }
  | { kind: "error"; message: string };

export async function getEvalSummary(): Promise<EvalSummaryResult> {
  try {
    const res = await fetch(`${API_BASE_URL}/eval/summary`, {
      cache: "no-store",
    });
    if (res.status === 404) return { kind: "not_computed" };
    if (!res.ok) {
      return { kind: "error", message: `Request failed with status ${res.status}` };
    }
    const data = (await res.json()) as EvalSummary;
    return { kind: "ok", data };
  } catch (err) {
    return {
      kind: "error",
      message: err instanceof Error ? err.message : "Network error",
    };
  }
}

export function analyzeStreamUrl(): string {
  return `${WS_BASE_URL}/analyze/stream`;
}

export function redteamStreamUrl(): string {
  return `${WS_BASE_URL}/redteam/stream`;
}
