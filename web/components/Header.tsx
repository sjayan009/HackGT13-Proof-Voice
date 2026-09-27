"use client";

import { useEffect, useState } from "react";
import { apiBase, isApiOverridden } from "@/lib/api";
import { useHealth } from "@/lib/useHealth";
import Icon from "./Icon";

function HealthPill() {
  const health = useHealth();
  // Read after mount so server and client markup match.
  const [backend, setBackend] = useState<{ url: string; custom: boolean } | null>(null);
  useEffect(() => setBackend({ url: apiBase(), custom: isApiOverridden() }), []);
  const API_BASE_URL = backend?.url ?? "the configured backend";

  let cls = "";
  let text = "Connecting…";
  let detail = "";
  let title = `ProofVoice API at ${API_BASE_URL}`;

  if (health.kind === "ok") {
    const d = health.data;
    cls = "ok";
    text = "API online";
    detail = [d.detector?.name, d.detector?.device?.toUpperCase()].filter(Boolean).join(" · ");
    title = `${title}\nDetector: ${d.detector?.name ?? "?"} ${d.detector?.version ?? ""} on ${
      d.detector?.device ?? "?"
    }\nGrok red team: ${d.grok_available ? "live xAI key" : "cached fixture only"}`;
  } else if (health.kind === "down") {
    cls = "down";
    text = "API offline";
    title = `${title} is unreachable (${health.message}). Retrying…`;
  }

  return (
    <span className={`health-pill ${cls}`} title={title} role="status" aria-live="polite">
      <span className="dot" aria-hidden />
      <span className="vh-sm">{text}</span>
      {backend?.custom && (
        <span className="badge badge-warn" title="Set with ?api= in the URL; use ?api=default to reset">
          custom
        </span>
      )}
      {detail && (
        <span className="detail" style={{ color: "var(--text-faint)" }}>
          {detail}
        </span>
      )}
    </span>
  );
}

export default function Header({ onOpenEval }: { onOpenEval: () => void }) {
  return (
    <header className="app-header">
      <div className="brand">
        <span className="brand-mark" aria-hidden>
          <Icon name="logo" size={20} strokeWidth={2.2} />
        </span>
        <div className="brand-text">
          <h1>ProofVoice</h1>
          <div className="tagline">A real-time evidence layer for synthetic speech</div>
        </div>
      </div>
      <div className="header-actions">
        <HealthPill />
        <button className="btn" onClick={onOpenEval} aria-haspopup="dialog">
          <Icon name="chart" size={15} />
          <span className="vh-sm">Evaluation</span>
        </button>
      </div>
    </header>
  );
}
