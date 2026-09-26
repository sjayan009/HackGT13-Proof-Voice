"use client";

import { useState } from "react";
import type { AnalysisReport } from "@/lib/types";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import ForensicLab from "@/components/ForensicLab";
import LiveTrust from "@/components/LiveTrust";
import RedTeam from "@/components/RedTeam";
import EvalDrawer from "@/components/EvalDrawer";

type Tab = "forensic" | "live" | "redteam";

export default function Home() {
  const [tab, setTab] = useState<Tab>("forensic");
  const [evalOpen, setEvalOpen] = useState(false);
  const [lastHumanReport, setLastHumanReport] = useState<AnalysisReport | null>(
    null
  );

  return (
    <div className="app-shell">
      <Header onOpenEval={() => setEvalOpen(true)} />

      <nav className="tabs">
        <button
          className={`tab-btn ${tab === "forensic" ? "active" : ""}`}
          onClick={() => setTab("forensic")}
        >
          Forensic Lab
        </button>
        <button
          className={`tab-btn ${tab === "live" ? "active" : ""}`}
          onClick={() => setTab("live")}
        >
          Live Trust
        </button>
        <button
          className={`tab-btn ${tab === "redteam" ? "active" : ""}`}
          onClick={() => setTab("redteam")}
        >
          Red Team (Grok)
        </button>
      </nav>

      <main className="main-content">
        {tab === "forensic" && <ForensicLab />}
        {tab === "live" && (
          <LiveTrust onReportReady={(r) => setLastHumanReport(r)} />
        )}
        {tab === "redteam" && <RedTeam lastHumanReport={lastHumanReport} />}
      </main>

      <Footer />

      {evalOpen && <EvalDrawer onClose={() => setEvalOpen(false)} />}
    </div>
  );
}
