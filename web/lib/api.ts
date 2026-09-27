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

const ENV_API = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
const ENV_WS = process.env.NEXT_PUBLIC_WS_BASE_URL?.replace(/\/+$/, "");
const OVERRIDE_KEY = "proofvoice.api";

function readOverride(): string | null {
  if (typeof window === "undefined") return null;
  try {
    // ?api=https://host points this browser at another backend (remembered); ?api=default clears it.
    const q = new URLSearchParams(window.location.search).get("api");
    if (q === "default") window.localStorage.removeItem(OVERRIDE_KEY);
    else if (q && /^https?:\/\/[^\s]+$/i.test(q)) window.localStorage.setItem(OVERRIDE_KEY, q.replace(/\/+$/, ""));
    return window.localStorage.getItem(OVERRIDE_KEY);
  } catch {
    return null;
  }
}

/** Backend base URL: a per-browser override (?api=…) wins over the build-time env. */
export function apiBase(): string {
  return readOverride() ?? ENV_API;
}

/** True when this browser is pointed at a backend other than the deployment default. */
export function isApiOverridden(): boolean {
  const o = readOverride();
  return !!o && o !== ENV_API;
}

export function wsBase(): string {
  const o = readOverride();
  if (!o && ENV_WS) return ENV_WS;
  return (o ?? ENV_API).replace(/^http/i, "ws");
}

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

/** User-facing message for any thrown request error (network, HTTP, abort). */
export function describeError(err: unknown, what = "the ProofVoice API"): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof DOMException && err.name === "TimeoutError")
    return `${what} did not respond in time.`;
  if (err instanceof TypeError)
    return `Could not reach ${what} at ${apiBase()}. Check that the backend is running.`;
  return err instanceof Error ? err.message : "Unknown error";
}

export function isAbort(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

function timeoutSignal(ms: number, outer?: AbortSignal): AbortSignal {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(new DOMException("timeout", "TimeoutError")), ms);
  outer?.addEventListener("abort", () => ctrl.abort(outer.reason), { once: true });
  ctrl.signal.addEventListener("abort", () => clearTimeout(t), { once: true });
  return ctrl.signal;
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${apiBase()}/health`, {
    cache: "no-store",
    signal: timeoutSignal(4000),
  });
  return parseJsonOrThrow<HealthResponse>(res);
}

export async function analyzeFile(file: File, signal?: AbortSignal): Promise<AnalysisReport> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${apiBase()}/analyze/file`, {
    method: "POST",
    body: form,
    signal: timeoutSignal(180_000, signal),
  });
  return parseJsonOrThrow<AnalysisReport>(res);
}

export async function redteamGrok(
  req: RedteamGrokRequest,
  signal?: AbortSignal
): Promise<RedteamGrokResponse> {
  const res = await fetch(`${apiBase()}/redteam/grok`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal: timeoutSignal(120_000, signal),
  });
  return parseJsonOrThrow<RedteamGrokResponse>(res);
}

export async function explainReport(
  report: AnalysisReport,
  signal?: AbortSignal
): Promise<ExplainResponse> {
  const body: ExplainRequest = { report };
  const res = await fetch(`${apiBase()}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: timeoutSignal(60_000, signal),
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
    const res = await fetch(`${apiBase()}/eval/summary`, {
      cache: "no-store",
      signal: timeoutSignal(10_000),
    });
    if (res.status === 404) return { kind: "not_computed" };
    if (!res.ok) {
      return { kind: "error", message: `Request failed with status ${res.status}` };
    }
    const data = (await res.json()) as EvalSummary;
    return { kind: "ok", data };
  } catch (err) {
    return { kind: "error", message: describeError(err) };
  }
}

export function analyzeStreamUrl(): string {
  return `${wsBase()}/analyze/stream`;
}

export function redteamStreamUrl(): string {
  return `${wsBase()}/redteam/stream`;
}
