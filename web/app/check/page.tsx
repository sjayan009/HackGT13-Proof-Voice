"use client";

import { useRef, useState } from "react";
import Link from "next/link";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import Callout from "@/components/Callout";
import Icon from "@/components/Icon";
import { analyzeFile, describeError } from "@/lib/api";
import type { AnalysisReport, Status } from "@/lib/types";

// Voice messages from WhatsApp/Telegram (.opus/.ogg), iPhone (.m4a/.caf), Android (.amr/.3gp/.m4a) and most apps.
const ACCEPT = "audio/*,video/*,.opus,.ogg,.oga,.m4a,.caf,.amr,.3gp,.aac,.mp3,.wav,.webm,.flac";

const VERDICT: Record<Status, { tone: "bad" | "warn" | "ok" | "none"; title: string; body: string }> = {
  likely_synthetic: {
    tone: "bad",
    title: "Signs of an AI-generated voice",
    body: "This recording shows strong patterns of synthetic speech. Don't act on it until you have checked with the person directly.",
  },
  inconclusive: {
    tone: "warn",
    title: "Can't tell either way",
    body: "Parts of it look synthetic and parts don't, or the audio quality makes it hard to be sure. Treat it with caution.",
  },
  likely_human: {
    tone: "ok",
    title: "No strong signs of an AI voice",
    body: "That does not prove who sent it: a real person can still be a scammer, and some new AI voices are not detected.",
  },
  insufficient_evidence: {
    tone: "none",
    title: "Not enough clear speech to check",
    body: "Try a longer message, or one with less background noise or music.",
  },
};

const sec = (ms: number) => `${(ms / 1000).toFixed(1)} s`;

export default function CheckPage() {
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<AnalysisReport | null>(null);

  const check = async (file: File | undefined) => {
    if (!file) return;
    setBusy(file.name);
    setError(null);
    setReport(null);
    try {
      setReport(await analyzeFile(file));
    } catch (e) {
      setError(describeError(e));
    } finally {
      setBusy(null);
      if (input.current) input.current.value = "";
    }
  };

  const v = report ? VERDICT[report.status] : null;
  const regions = report?.suspicious_regions.filter((r) => r.end_ms > r.start_ms) ?? [];

  return (
    <div className="app-shell">
      <div className="chrome">
        <Header />
      </div>
      <main className="main-content check stack" id="main" tabIndex={-1}>
        <div className="page-intro">
          <h2>Check a voice message</h2>
          <p>
            Got a voice message that sounds like someone you know asking for money or help? Check it for signs of an
            AI-cloned voice. Your audio is analysed and deleted straight away; nothing is stored.
          </p>
        </div>

        <input
          ref={input}
          type="file"
          accept={ACCEPT}
          hidden
          onChange={(e) => check(e.target.files?.[0])}
        />
        <button
          type="button"
          className={`dropzone ${drag ? "active" : ""}`}
          disabled={busy !== null}
          onClick={() => input.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            check(e.dataTransfer.files?.[0]);
          }}
        >
          <span className="dropzone-icon" aria-hidden>
            <Icon name="upload" size={24} />
          </span>
          <span className="dropzone-title">{busy ? `Checking ${busy}…` : "Choose a voice message"}</span>
          <span className="dropzone-hint">WhatsApp, Telegram, iMessage, voicemail, or any audio file</span>
        </button>

        <details className="check-howto">
          <summary>How do I get the voice message out of my app?</summary>
          <ul>
            <li>
              <b>WhatsApp on iPhone:</b> press and hold the voice message, tap <i>Forward</i>, then the share icon, then{" "}
              <i>Save to Files</i>. Choose it from Files here.
            </li>
            <li>
              <b>WhatsApp on Android:</b> press and hold the voice message, tap the share icon, and save it to your files
              or Drive.
            </li>
            <li>
              <b>iMessage:</b> press and hold the audio message and tap <i>Save</i>; it goes to Voice Memos, where you can
              share it to Files.
            </li>
            <li>
              <b>Telegram:</b> press and hold the voice message and choose <i>Save to Downloads</i> or <i>Share</i>.
            </li>
            <li>
              <b>On a computer:</b> drag the file onto the box above.
            </li>
          </ul>
        </details>

        {busy && (
          <div className="panel appear" role="status" aria-live="polite">
            <div className="progress-card">
              <span className="spinner" aria-hidden />
              <span>Listening for signs of synthetic speech… this takes a few seconds.</span>
            </div>
          </div>
        )}

        {error && (
          <Callout tone="error" title="Couldn't check that file">
            {error}
          </Callout>
        )}

        {report && v && (
          <section className={`panel appear check-verdict tone-${v.tone}`} aria-live="polite">
            <h3>{v.title}</h3>
            <p>{v.body}</p>
            {regions.length > 0 && report.status !== "likely_human" && (
              <p className="panel-sub">
                Parts that looked synthetic:{" "}
                {regions.map((r) => `${sec(r.start_ms)}–${sec(r.end_ms)}`).join(", ")}
              </p>
            )}
            <p className="panel-sub">
              {report.file.name}
              {report.file.duration_s != null && ` · ${report.file.duration_s.toFixed(1)} s checked`}
            </p>
          </section>
        )}

        <section className="panel check-todo">
          <h3>Before you send money or share anything</h3>
          <ul>
            <li>
              <Icon name="check" size={16} /> Call the person back on a number you already have, not the one that
              messaged you.
            </li>
            <li>
              <Icon name="check" size={16} /> Ask something only they would know, or agree a family code word.
            </li>
            <li>
              <Icon name="check" size={16} /> Never send money, gift cards, crypto or verification codes because of a
              voice message alone.
            </li>
            <li>
              <Icon name="check" size={16} /> In the US, report scams at{" "}
              <a href="https://reportfraud.ftc.gov" target="_blank" rel="noreferrer">
                reportfraud.ftc.gov
              </a>
              .
            </li>
          </ul>
        </section>

        <p className="panel-sub">
          ProofVoice gives evidence, not proof. It has been tested on research data and simulated phone audio, not yet
          on a large set of real voice messages, so it can be wrong in both directions. Want the technical breakdown?{" "}
          <Link href="/">Open the file in the Lab</Link>.
        </p>
      </main>
      <Footer />
    </div>
  );
}
