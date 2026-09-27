// Display formatting only — never changes the underlying numbers.

import type { AnalysisReport, Status } from "./types";

export const STATUS_LABEL: Record<Status, string> = {
  likely_human: "Likely human",
  inconclusive: "Inconclusive",
  likely_synthetic: "Likely synthetic",
  insufficient_evidence: "Insufficient evidence",
};

export const STATUS_COLOR: Record<Status, string> = {
  likely_human: "var(--green)",
  inconclusive: "var(--amber)",
  likely_synthetic: "var(--red)",
  insufficient_evidence: "var(--gray)",
};

export function isFiniteNumber(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

export function pct(p: number | null | undefined, digits = 0): string {
  return isFiniteNumber(p) ? `${(p * 100).toFixed(digits)}%` : "—";
}

export function ms(v: number | null | undefined): string {
  if (!isFiniteNumber(v)) return "—";
  if (v >= 10_000) return `${(v / 1000).toFixed(1)} s`;
  return `${Math.round(v).toLocaleString()} ms`;
}

export function seconds(msValue: number): string {
  const s = msValue / 1000;
  if (s >= 60) return `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
  return `${Number.isInteger(s) || s >= 10 ? Math.round(s) : s.toFixed(1)}s`;
}

/** Round axis ticks (0, 0.5 s, 1 s, 2 s, 5 s, …) that fit roughly `maxTicks` labels across `spanMs`. */
export function timeTicks(spanMs: number, maxTicks: number): number[] {
  const steps = [250, 500, 1000, 2000, 5000, 10_000, 15_000, 30_000, 60_000, 120_000, 300_000];
  const step = steps.find((s) => spanMs / s <= maxTicks) ?? Math.ceil(spanMs / maxTicks / 60_000) * 60_000;
  const out: number[] = [];
  for (let t = 0; t <= spanMs + 1e-6; t += step) out.push(t);
  return out;
}

export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/** Compact number for tables: integers as-is, otherwise 4 significant digits. */
export function num(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return String(v);
    if (Number.isInteger(v)) return v.toLocaleString();
    const abs = Math.abs(v);
    if (abs !== 0 && (abs < 1e-3 || abs >= 1e6)) return v.toExponential(2);
    return Number(v.toPrecision(4)).toString();
  }
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

export function humanizeKey(k: string): string {
  return k.replace(/_/g, " ");
}

export function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

/** Mirrors the backend's status bands (api/app/orchestration/pipeline.py: status_for). */
export function decisionBands(threshold: number) {
  return { human: threshold * 0.5, synthetic: Math.max(threshold, threshold + (1 - threshold) * 0.35) };
}

/** Probability of the side the evidence leans toward, e.g. p=0.003 → "99.7%" human. */
export function leaning(p: number | null | undefined): { text: string; side: "human" | "synthetic" } | null {
  if (!isFiniteNumber(p)) return null;
  const side = p < 0.5 ? "human" : "synthetic";
  const v = (side === "human" ? 1 - p : p) * 100;
  const text = v >= 99.95 ? ">99.9%" : v >= 99 ? `${v.toFixed(1)}%` : `${Math.round(v)}%`;
  return { text, side };
}

/** One plain-language sentence restating fields already in the report (no inference). */
export function verdictSentence(r: AnalysisReport): string {
  const l = leaning(r.synthetic_probability);
  const reading = l ? `${l.text} ${l.side}` : "unavailable";
  const conf = pct(r.analysis_confidence);
  const regions = r.suspicious_regions?.length ?? 0;
  const regionText =
    regions === 0
      ? "No windows were flagged as suspicious."
      : `${regions} suspicious region${regions === 1 ? " was" : "s were"} flagged on the timeline.`;
  switch (r.status) {
    case "insufficient_evidence":
      return `Not enough usable speech to reach a decision (analysis confidence ${conf}). ${regionText}`;
    case "inconclusive":
      return `The evidence reads ${reading}, which falls in the inconclusive zone: not strong enough either way, so it should go to a human analyst. ${regionText}`;
    default:
      return `The evidence reads ${reading} at ${conf} analysis confidence. ${regionText}`;
  }
}

export function downloadJson(data: unknown, filename: string) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function reportFilename(r: AnalysisReport): string {
  const base = (r.file?.name || "stream").replace(/\.[^.]+$/, "").replace(/[^\w.-]+/g, "_");
  return `proofvoice_${base}_${(r.id || "report").slice(0, 8)}.json`;
}
