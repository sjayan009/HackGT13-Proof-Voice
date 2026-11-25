"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { apiBase, isApiOverridden } from "@/lib/api";
import { useHealth } from "@/lib/useHealth";
import Icon from "./Icon";
import ThemeSwitch from "./ThemeSwitch";

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

const NAV = [
  { href: "/", label: "Lab" },
  { href: "/check", label: "Check" },
  { href: "/guardian", label: "Guardian" },
  { href: "/engine", label: "Control Center" },
];

function SiteNav() {
  const path = usePathname();
  return (
    <nav className="site-nav" aria-label="Sections">
      {NAV.map((n) => (
        <Link key={n.href} href={n.href} aria-current={path === n.href ? "page" : undefined}>
          {n.label}
        </Link>
      ))}
    </nav>
  );
}

export default function Header({ onOpenEval }: { onOpenEval?: () => void }) {
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
      <SiteNav />
      <div className="header-actions">
        <HealthPill />
        <ThemeSwitch />
        {onOpenEval && (
          <button className="btn" onClick={onOpenEval} aria-haspopup="dialog">
            <Icon name="chart" size={15} />
            <span className="vh-sm">Evaluation</span>
          </button>
        )}
      </div>
    </header>
  );
}
