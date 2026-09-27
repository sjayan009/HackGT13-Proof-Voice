"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { AnalysisReport } from "@/lib/types";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import ForensicLab from "@/components/ForensicLab";
import LiveTrust from "@/components/LiveTrust";
import RedTeam from "@/components/RedTeam";
import EvalDrawer from "@/components/EvalDrawer";
import Icon from "@/components/Icon";

type Tab = "forensic" | "live" | "redteam";

const TABS: { id: Tab; label: string; icon: "file" | "mic" | "spark" }[] = [
  { id: "forensic", label: "Forensic Lab", icon: "file" },
  { id: "live", label: "Live Trust", icon: "mic" },
  { id: "redteam", label: "Red Team", icon: "spark" },
];

function tabFromHash(): Tab | null {
  if (typeof window === "undefined") return null;
  const h = window.location.hash.replace("#", "");
  return TABS.some((t) => t.id === h) ? (h as Tab) : null;
}

export default function Home() {
  const [tab, setTab] = useState<Tab>("forensic");
  const [evalOpen, setEvalOpen] = useState(false);
  const [lastHumanReport, setLastHumanReport] = useState<AnalysisReport | null>(null);
  const tabRefs = useRef<Record<Tab, HTMLButtonElement | null>>({
    forensic: null,
    live: null,
    redteam: null,
  });

  // Deep-linkable tabs (#live, #redteam); back/forward keeps working.
  useEffect(() => {
    const sync = () => {
      const t = tabFromHash();
      if (t) setTab(t);
    };
    sync();
    window.addEventListener("hashchange", sync);
    return () => window.removeEventListener("hashchange", sync);
  }, []);

  const select = useCallback((t: Tab, focus = false) => {
    setTab(t);
    if (window.location.hash !== `#${t}`) {
      history.replaceState(history.state, "", t === "forensic" ? window.location.pathname : `#${t}`);
    }
    if (focus) tabRefs.current[t]?.focus();
  }, []);

  const onTabKeyDown = (e: React.KeyboardEvent) => {
    const i = TABS.findIndex((t) => t.id === tab);
    let next = -1;
    if (e.key === "ArrowRight") next = (i + 1) % TABS.length;
    else if (e.key === "ArrowLeft") next = (i - 1 + TABS.length) % TABS.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = TABS.length - 1;
    if (next >= 0) {
      e.preventDefault();
      select(TABS[next].id, true);
    }
  };

  const index = TABS.findIndex((t) => t.id === tab);

  return (
    <div className="app-shell">
      <div className="chrome">
        <Header onOpenEval={() => setEvalOpen(true)} />
        <div className="nav-row">
          <div className="segmented" role="tablist" aria-label="Mode" onKeyDown={onTabKeyDown}>
            <span
              className="segmented-indicator"
              aria-hidden
              style={{
                width: `calc((100% - 6px) / ${TABS.length})`,
                transform: `translateX(${index * 100}%)`,
              }}
            />
            {TABS.map((t) => (
              <button
                key={t.id}
                ref={(el) => {
                  tabRefs.current[t.id] = el;
                }}
                role="tab"
                id={`tab-${t.id}`}
                aria-selected={tab === t.id}
                aria-controls={`panel-${t.id}`}
                tabIndex={tab === t.id ? 0 : -1}
                className="segment"
                onClick={() => select(t.id)}
              >
                <Icon name={t.icon} size={14} />
                {t.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <main className="main-content" id="main" tabIndex={-1}>
        {/* Panels stay mounted so results and running sessions survive tab switches. */}
        <section
          role="tabpanel"
          id="panel-forensic"
          aria-labelledby="tab-forensic"
          hidden={tab !== "forensic"}
          className="tab-panel"
        >
          <ForensicLab />
        </section>
        <section
          role="tabpanel"
          id="panel-live"
          aria-labelledby="tab-live"
          hidden={tab !== "live"}
          className="tab-panel"
        >
          <LiveTrust onReportReady={setLastHumanReport} />
        </section>
        <section
          role="tabpanel"
          id="panel-redteam"
          aria-labelledby="tab-redteam"
          hidden={tab !== "redteam"}
          className="tab-panel"
        >
          <RedTeam lastHumanReport={lastHumanReport} onGoLive={() => select("live")} />
        </section>
      </main>

      <Footer />

      <EvalDrawer open={evalOpen} onClose={() => setEvalOpen(false)} />
    </div>
  );
}
