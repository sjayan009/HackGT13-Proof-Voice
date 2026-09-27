// Demo sample library (web/public/samples, built by scripts/build-samples.py from held-out clips).

import type { AnalysisReport } from "./types";

export interface Sample {
  id: string;
  title: string;
  group: string;
  truth: "bonafide" | "spoof";
  file: string;
  report: string;
  duration_s: number | null;
  bytes: number;
  transcript: string | null;
  note: string;
  credit: string;
  held_out: boolean;
}

export interface SampleManifest {
  generated_at: string;
  detector: { name?: string; version?: string; device?: string };
  samples: Sample[];
}

let manifestPromise: Promise<SampleManifest | null> | null = null;

export function loadSampleManifest(): Promise<SampleManifest | null> {
  manifestPromise ??= fetch("/samples/manifest.json", { cache: "no-cache" })
    .then((r) => (r.ok ? (r.json() as Promise<SampleManifest>) : null))
    .catch(() => null);
  return manifestPromise;
}

export async function fetchSampleFile(s: Sample, signal?: AbortSignal): Promise<File> {
  const res = await fetch(s.file, { signal });
  if (!res.ok) throw new Error(`Could not load sample audio (${res.status}).`);
  const blob = await res.blob();
  const name = s.file.split("/").pop() || `${s.id}.wav`;
  return new File([blob], name, { type: blob.type || "audio/wav" });
}

/** Reference report computed by the real API when the library was built. */
export async function fetchSampleReport(s: Sample): Promise<AnalysisReport> {
  const res = await fetch(s.report);
  if (!res.ok) throw new Error(`Could not load the reference report (${res.status}).`);
  return (await res.json()) as AnalysisReport;
}

export function groupSamples(samples: Sample[]): [string, Sample[]][] {
  const groups = new Map<string, Sample[]>();
  for (const s of samples) {
    if (!groups.has(s.group)) groups.set(s.group, []);
    groups.get(s.group)!.push(s);
  }
  return Array.from(groups.entries());
}

export function truthLabel(t: Sample["truth"]) {
  return t === "bonafide" ? "Real" : "Synthetic";
}
