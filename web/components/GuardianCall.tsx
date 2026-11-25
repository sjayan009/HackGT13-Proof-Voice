"use client";

// One live (or just-ended) Guardian call: PRD §3 call card, evidence meter, verification workflow,
// evidence timeline. Everything shown comes from WS /guardian/live events; nothing is inferred locally.

import { useEffect, useState } from "react";
import type {
  CallAlertEvent,
  CallEndedEvent,
  CallStartedEvent,
  CallUpdateEvent,
  RiskState,
  TrustedContact,
  VerificationOutcome,
} from "@/lib/types";
import { describeError, recordVerification } from "@/lib/api";
import { isFiniteNumber, seconds } from "@/lib/format";
import GuardianTimeline, { type TimelinePoint } from "./GuardianTimeline";
import Callout from "./Callout";
import Icon from "./Icon";

export interface LiveSession {
  id: string;
  started?: CallStartedEvent;
  updates: CallUpdateEvent[];
  alert?: CallAlertEvent;
  ended?: CallEndedEvent;
}

export const STATE_COPY: Record<RiskState, { title: string; body: string; tone: "ok" | "warn" | "bad" | "off" }> = {
  no_suspicious_evidence: {
    title: "No suspicious voice patterns so far",
    body: "There has been enough clear speech and no strong sign of a synthetic voice. This is not proof of who is calling. If they ask for money or private information, still check with them on a number you already know.",
    tone: "ok",
  },
  inconclusive: {
    title: "Not enough to tell yet",
    body: "There is too little clear speech, the line is poor, or the signals are mixed. Be careful with any request for money or private information.",
    tone: "warn",
  },
  suspicious: {
    title: "Potential synthetic speech detected",
    body: "Suspicious voice patterns were detected. Verify independently before sharing information or sending money.",
    tone: "bad",
  },
  analysis_unavailable: {
    title: "Analysis unavailable",
    body: "Your call continues normally, but it is not being checked right now.",
    tone: "off",
  },
};

const OUTCOMES: { value: VerificationOutcome; label: string }[] = [
  { value: "verified_genuine", label: "I reached them on their known number and they confirmed the request" },
  { value: "confirmed_scam", label: "I reached them and they did not make this request" },
  { value: "could_not_reach", label: "I could not reach them" },
  { value: "skipped", label: "Skip for now" },
];

export const OUTCOME_LABEL: Record<VerificationOutcome, string> = {
  verified_genuine: "Request confirmed with the person directly",
  confirmed_scam: "The person said they did not make the request",
  could_not_reach: "Could not reach the person",
  skipped: "Verification skipped",
};

export function isSimulated(s: { consent?: string | null }) {
  return s.consent === "simulated";
}

function clock(msValue: number) {
  const s = Math.max(0, Math.floor(msValue / 1000));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

export function EvidenceMeter({ evidence }: { evidence: number | null }) {
  const v = isFiniteNumber(evidence) ? Math.max(0, Math.min(1.5, evidence)) : null;
  return (
    <div>
      <div
        className="evidence-meter-track"
        role="meter"
        aria-label="Evidence of synthetic speech (alert at 1.0)"
        aria-valuemin={0}
        aria-valuemax={1.5}
        aria-valuenow={v ?? undefined}
        aria-valuetext={v == null ? "No evidence reading yet" : `${v.toFixed(2)} of 1.5; alert line at 1.0`}
      >
        <div
          className={`evidence-meter-fill ${v != null && v >= 1 ? "over" : ""}`}
          style={{ transform: `scaleX(${(v ?? 0) / 1.5})` }}
        />
        <div className="evidence-meter-line" style={{ left: `${(1 / 1.5) * 100}%` }} aria-hidden />
      </div>
      <div className="evidence-meter-labels">
        <span>Evidence: {v == null ? "no reading yet" : v.toFixed(2)}</span>
        <span>Alert line at 1.0</span>
      </div>
    </div>
  );
}

function useNow(active: boolean) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [active]);
  return now;
}

function Verification({
  sessionId,
  contacts,
  contactsNote,
  token,
}: {
  sessionId: string;
  contacts: TrustedContact[];
  contactsNote: string | null;
  token: string | null;
}) {
  const [contactId, setContactId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState<VerificationOutcome | null>(null);
  const [error, setError] = useState<string | null>(null);

  const submit = async (outcome: VerificationOutcome) => {
    setBusy(true);
    setError(null);
    try {
      await recordVerification(sessionId, { contact_id: contactId, outcome }, token);
      setDone(outcome);
    } catch (e) {
      setError(describeError(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="panel stack" aria-label="Verify independently">
      <div>
        <h3>Verify independently</h3>
        <p className="panel-sub" style={{ margin: "6px 0 0" }}>
          Hang up or put this call on hold, then call the person yourself on a number you already have for them. Do
          not use a number the caller gives you.
        </p>
      </div>
      {contacts.length > 0 ? (
        <ul className="contact-list">
          {contacts.map((c) => (
            <li key={c.id} className={contactId === c.id ? "selected" : undefined}>
              <span>
                <strong>{c.name}</strong>
                {c.relationship && <span style={{ color: "var(--text-dim)" }}> · {c.relationship}</span>}
                <br />
                <span className="mono">{c.phone}</span>
              </span>
              <a className="btn btn-primary" href={`tel:${c.phone}`} onClick={() => setContactId(c.id)}>
                Call {c.name}
              </a>
            </li>
          ))}
        </ul>
      ) : (
        <Callout tone="info">{contactsNote ?? "You have no trusted contacts yet. Add them below so they appear here."}</Callout>
      )}
      {done ? (
        <Callout tone="info" title="Recorded">
          {OUTCOME_LABEL[done]}. Your answer was saved.
        </Callout>
      ) : (
        <div>
          <p style={{ margin: "0 0 8px" }}>What happened?</p>
          <div className="outcome-grid">
            {OUTCOMES.map((o) => (
              <button key={o.value} className="btn" disabled={busy} onClick={() => submit(o.value)}>
                {o.label}
              </button>
            ))}
          </div>
        </div>
      )}
      {error && <Callout tone="error" title="Could not save your answer">{error}</Callout>}
    </section>
  );
}

export function updatesToPoints(updates: CallUpdateEvent[]): TimelinePoint[] {
  return updates.map((u) => ({
    t_ms: u.t_ms,
    evidence: u.evidence,
    p: u.window?.synthetic_probability ?? null,
    usable: u.window ? u.window.usable : null,
    skipped: u.window == null,
    action: u.compute?.action,
  }));
}

export default function GuardianCall({
  session,
  contacts,
  contactsNote,
  token,
  onDismiss,
}: {
  session: LiveSession;
  contacts: TrustedContact[];
  contactsNote: string | null;
  token: string | null;
  onDismiss: () => void;
}) {
  const { started, updates, alert, ended } = session;
  const last = updates[updates.length - 1];
  const live = !ended;
  const now = useNow(live);
  const [showVerify, setShowVerify] = useState(false);
  const [showEvidence, setShowEvidence] = useState(false);

  useEffect(() => {
    if (alert) setShowVerify(true);
  }, [alert]);

  const state: RiskState | null =
    ended?.final_state ?? last?.state ?? (started?.consent === "declined" ? "analysis_unavailable" : null);
  const copy = state ? STATE_COPY[state] : null;
  const evidence = last?.evidence ?? (ended ? ended.peak_evidence : null);

  const elapsed = ended
    ? ended.duration_ms
    : started
      ? now - Date.parse(started.started_at)
      : (last?.t_ms ?? 0);

  const strip = updates.slice(-48);
  const simulated = started ? isSimulated(started) : false;

  return (
    <article className={`panel call-card stack appear ${state === "suspicious" ? "tone-suspicious" : ""}`} aria-live="polite">
      <div className="call-top">
        <div>
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <span className={`live-indicator ${live ? "on" : ""}`}>
              <span className="dot" aria-hidden />
              {live ? "Monitoring" : "Call ended"}
            </span>
            {simulated && (
              <span className="badge badge-warn">
                Simulation{started?.scenario ? ` · ${started.scenario} sample` : ""}
              </span>
            )}
          </div>
          <p className="call-caller">
            {started ? (started.caller ? `Incoming call · ${started.caller}` : "Incoming call · Unknown caller") : "Call in progress"}
          </p>
          {!started && <p className="panel-sub" style={{ margin: 0 }}>Joined after the call started; caller details were not received.</p>}
        </div>
        <div style={{ textAlign: "right" }}>
          <div className="call-timer" aria-label={`Call time ${clock(elapsed)}`}>
            {clock(elapsed)}
          </div>
          {!live && (
            <button className="btn btn-ghost btn-sm" onClick={onDismiss}>
              Dismiss
            </button>
          )}
        </div>
      </div>

      <div className="level-strip" aria-hidden>
        {strip.length === 0
          ? null
          : strip.map((u, i) => {
              const db = u.window?.level_dbfs;
              const h = isFiniteNumber(db) ? Math.max(0.08, Math.min(1, (db + 60) / 60)) : 0.08;
              return <i key={i} className={u.window ? undefined : "skipped"} style={{ height: `${h * 100}%` }} />;
            })}
      </div>

      {alert && (
        <div className="alert-banner" role="alert">
          <Icon name="alert" size={26} strokeWidth={2.2} />
          <div>
            {alert.message}
            <small>
              Alert raised at {seconds(alert.t_ms)} of caller audio.
              {alert.sms_sent ? " A text message alert was also sent to your phone." : ""}
            </small>
          </div>
        </div>
      )}

      {copy ? (
        <div className={`risk-state tone-${copy.tone}`}>
          <h3 className="risk-title">{copy.title}</h3>
          <p>{copy.body}</p>
          {last?.reason && <p className="risk-reason">Why: {last.reason}</p>}
          {started?.consent === "declined" && (
            <p className="risk-reason">The caller did not agree to the check, so this call is not being analysed.</p>
          )}
        </div>
      ) : (
        <div className="risk-state">
          <h3 className="risk-title">Listening…</h3>
          <p>The first check needs a few seconds of the caller&apos;s speech.</p>
        </div>
      )}

      <EvidenceMeter evidence={evidence} />

      <div className="verdict-actions">
        <button className="btn btn-primary" onClick={() => setShowVerify((v) => !v)} aria-expanded={showVerify}>
          <Icon name="check" size={16} /> Verify contact
        </button>
        <button className="btn" onClick={() => setShowEvidence((v) => !v)} aria-expanded={showEvidence}>
          <Icon name="chart" size={16} /> {showEvidence ? "Hide evidence" : "View evidence"}
        </button>
      </div>

      {showVerify && <Verification sessionId={session.id} contacts={contacts} contactsNote={contactsNote} token={token} />}

      {showEvidence && (
        <section className="panel" aria-label="Evidence timeline">
          <div className="panel-head">
            <h3>Evidence over time</h3>
            <span className="panel-sub">One step every 0.5 s of caller audio</span>
          </div>
          <GuardianTimeline points={updatesToPoints(updates)} />
        </section>
      )}

      {ended && (
        <dl className="kv-list" style={{ margin: 0 }}>
          <dt className="k">Call length</dt>
          <dd className="v" style={{ margin: 0 }}>{clock(ended.duration_ms)}</dd>
          <dt className="k">Clear speech analysed</dt>
          <dd className="v" style={{ margin: 0 }}>{ended.usable_speech_s.toFixed(1)} s</dd>
          <dt className="k">Windows checked / skipped</dt>
          <dd className="v" style={{ margin: 0 }}>
            {ended.windows_inferred} / {ended.windows_skipped}
          </dd>
          <dt className="k">Highest evidence</dt>
          <dd className="v" style={{ margin: 0 }}>{ended.peak_evidence.toFixed(2)}</dd>
        </dl>
      )}

      <p className="panel-sub" style={{ margin: 0 }}>
        Guardian gives probabilistic evidence, not proof. It never decides who is calling.
      </p>
    </article>
  );
}
