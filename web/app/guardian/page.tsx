"use client";

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import type { Session } from "@supabase/supabase-js";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import Callout from "@/components/Callout";
import Icon from "@/components/Icon";
import GuardianCall, { type LiveSession } from "@/components/GuardianCall";
import { CallHistory, RowsEditor, useOwnerRows } from "@/components/GuardianAccount";
import { describeError, guardianLiveUrl, simulateCall } from "@/lib/api";
import { supabase } from "@/lib/supabase";
import type { GuardianLine, GuardianLiveEvent, Scenario, TrustedContact } from "@/lib/types";

// ---------------- live sessions (WS /guardian/live) ----------------

type Action = GuardianLiveEvent | { type: "dismiss"; id: string };

function reducer(state: LiveSession[], e: Action): LiveSession[] {
  if (e.type === "hello") return state;
  const id = e.type === "dismiss" ? e.id : e.session_id;
  if (e.type === "dismiss") return state.filter((s) => s.id !== id);
  const cur = state.find((s) => s.id === id) ?? { id, updates: [] };
  const rest = state.filter((s) => s.id !== id);
  let next: LiveSession;
  if (e.type === "call.started") next = { ...cur, started: e };
  else if (e.type === "call.update") next = { ...cur, updates: [...cur.updates, e] };
  else if (e.type === "call.alert") next = { ...cur, alert: e };
  else next = { ...cur, ended: e };
  // Existing cards keep their position; a new call goes on top.
  return state.some((s) => s.id === id) ? state.map((s) => (s.id === id ? next : s)) : [next, ...rest];
}

type Conn =
  | { kind: "connecting" }
  | { kind: "open"; devMode: boolean | null }
  | { kind: "retry"; inS: number; why: string }
  | { kind: "rejected" };

async function currentToken(): Promise<string | null> {
  if (!supabase) return null;
  const { data } = await supabase.auth.getSession();
  return data.session?.access_token ?? null;
}

function useGuardianLive(enabled: boolean, onEvent: (e: GuardianLiveEvent) => void) {
  const [conn, setConn] = useState<Conn>({ kind: "connecting" });
  const handler = useRef(onEvent);
  handler.current = onEvent;

  useEffect(() => {
    if (!enabled) return;
    let ws: WebSocket | null = null;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let attempt = 0;
    let stopped = false;

    const connect = async () => {
      setConn({ kind: "connecting" });
      const token = await currentToken(); // re-read each time so refreshed tokens are used
      if (stopped) return;
      ws = new WebSocket(guardianLiveUrl(token));
      ws.onopen = () => {
        attempt = 0;
        setConn({ kind: "open", devMode: null });
      };
      ws.onmessage = (m) => {
        let e: GuardianLiveEvent | { type: "error"; detail: string };
        try {
          e = JSON.parse(m.data as string);
        } catch {
          return;
        }
        if (e.type === "error") return; // followed by close 1008 when the token is rejected
        if (e.type === "hello") setConn({ kind: "open", devMode: e.dev_mode });
        handler.current(e);
      };
      ws.onclose = (ev) => {
        if (stopped) return;
        if (ev.code === 1008) {
          // Token rejected: don't retry; drop the local session so the page shows sign-in.
          setConn({ kind: "rejected" });
          supabase?.auth.signOut({ scope: "local" });
          return;
        }
        const delay = Math.min(30, 2 ** attempt);
        attempt++;
        setConn({ kind: "retry", inS: delay, why: ev.reason || (ev.code === 1006 ? "Connection lost" : `Closed (${ev.code})`) });
        timer = setTimeout(connect, delay * 1000);
      };
    };
    connect();
    return () => {
      stopped = true;
      clearTimeout(timer);
      ws?.close();
    };
  }, [enabled]);

  return conn;
}

function ConnPill({ conn }: { conn: Conn }) {
  const text =
    conn.kind === "open"
      ? `Connected${conn.devMode ? " (dev mode, no sign-in)" : ""}`
      : conn.kind === "connecting"
        ? "Connecting…"
        : conn.kind === "rejected"
          ? "Sign-in required: the server rejected this session"
          : `${conn.why}. Reconnecting in ${conn.inS} s`;
  return (
    <span className={`conn-pill ${conn.kind}`} role="status" aria-live="polite">
      <span className="dot" aria-hidden /> {text}
    </span>
  );
}

// ---------------- notifications ----------------

type Perm = NotificationPermission | "unsupported";

function useNotificationPermission(): [Perm, () => void] {
  const [perm, setPerm] = useState<Perm>("default");
  useEffect(() => setPerm(typeof Notification === "undefined" ? "unsupported" : Notification.permission), []);
  // Must be called from a click: browsers ignore permission prompts without a user gesture.
  const ask = () => {
    if (typeof Notification !== "undefined") Notification.requestPermission().then(setPerm);
  };
  return [perm, ask];
}

// ---------------- sign-in ----------------

function AuthPanel() {
  const [mode, setMode] = useState<"in" | "up">("in");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ tone: "error" | "info"; text: string } | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!supabase) return;
    setBusy(true);
    setMsg(null);
    const res =
      mode === "in"
        ? await supabase.auth.signInWithPassword({ email, password })
        : await supabase.auth.signUp({ email, password });
    setBusy(false);
    if (res.error) setMsg({ tone: "error", text: res.error.message });
    else if (mode === "up" && !res.data.session)
      setMsg({ tone: "info", text: "Check your email to confirm your account, then sign in." });
  };

  return (
    <section className="panel stack" style={{ maxWidth: 480 }}>
      <h3>{mode === "in" ? "Sign in to Guardian" : "Create your Guardian account"}</h3>
      <form onSubmit={submit} className="stack">
        <div>
          <label htmlFor="auth-email">Email</label>
          <input
            id="auth-email"
            className="text-input"
            style={{ width: "100%" }}
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>
        <div>
          <label htmlFor="auth-password">Password</label>
          <input
            id="auth-password"
            className="text-input"
            style={{ width: "100%" }}
            type="password"
            autoComplete={mode === "in" ? "current-password" : "new-password"}
            minLength={6}
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </div>
        <button className="btn btn-primary btn-lg" type="submit" disabled={busy}>
          {mode === "in" ? "Sign in" : "Create account"}
        </button>
      </form>
      <button className="link-btn" onClick={() => setMode(mode === "in" ? "up" : "in")}>
        {mode === "in" ? "New here? Create an account" : "Already have an account? Sign in"}
      </button>
      {msg && <Callout tone={msg.tone}>{msg.text}</Callout>}
    </section>
  );
}

// ---------------- page ----------------

const SCENARIOS: { id: Scenario; label: string }[] = [
  { id: "genuine", label: "Real voice sample" },
  { id: "synthetic", label: "Synthetic voice sample" },
  { id: "mixed", label: "Mixed sample" },
];

export default function GuardianPage() {
  const [authReady, setAuthReady] = useState(!supabase);
  const [session, setSession] = useState<Session | null>(null);

  useEffect(() => {
    if (!supabase) return;
    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session);
      setAuthReady(true);
    });
    const { data } = supabase.auth.onAuthStateChange((_e, s) => setSession(s));
    return () => data.subscription.unsubscribe();
  }, []);

  const signedIn = !!session;
  const devMode = !supabase;
  const canRun = devMode || signedIn;
  const userId = session?.user.id ?? null;

  const [sessions, dispatch] = useReducer(reducer, []);
  const [perm, askPerm] = useNotificationPermission();
  const [historyKey, setHistoryKey] = useState(0);

  const onEvent = useCallback((e: GuardianLiveEvent) => {
    dispatch(e);
    if (e.type === "call.alert" && typeof Notification !== "undefined" && Notification.permission === "granted") {
      new Notification("ProofVoice Guardian", { body: e.message, tag: e.session_id, requireInteraction: true });
    }
    if (e.type === "call.ended") setHistoryKey((k) => k + 1);
  }, []);

  const conn = useGuardianLive(canRun && authReady, onEvent);

  const contacts = useOwnerRows<TrustedContact>("trusted_contacts", signedIn);
  const lines = useOwnerRows<GuardianLine>("guardian_lines", signedIn);

  const [simBusy, setSimBusy] = useState<Scenario | null>(null);
  const [simError, setSimError] = useState<string | null>(null);
  const simulate = async (scenario: Scenario) => {
    setSimBusy(scenario);
    setSimError(null);
    try {
      await simulateCall({ scenario }, session?.access_token ?? null);
    } catch (e) {
      setSimError(describeError(e));
    } finally {
      setSimBusy(null);
    }
  };

  const contactsNote = devMode
    ? "Trusted contacts are saved to your account, and sign-in is not configured here (dev mode). Call the person on a number you already know."
    : null;

  return (
    <div className="app-shell">
      <div className="chrome">
        <Header />
      </div>
      <main className="main-content guardian stack" id="main" tabIndex={-1}>
        <div className="page-intro">
          <h2>ProofVoice Guardian</h2>
          <p>
            Guardian listens to calls on your protected line (only after the caller agrees) and warns you if the voice
            shows signs of being synthetic. It never hangs up, and it can&apos;t tell you who is really calling, so always
            check with people on a number you already know before sending money or sharing information.
          </p>
        </div>

        {!authReady ? (
          <div className="empty-state">Loading…</div>
        ) : !canRun ? (
          <AuthPanel />
        ) : (
          <>
            <div className="control-bar">
              <ConnPill conn={conn} />
              <div className="control-group">
                {perm === "default" && (
                  <button className="btn" onClick={askPerm}>
                    <Icon name="alert" size={16} /> Turn on desktop alerts
                  </button>
                )}
                {perm === "granted" && <span className="panel-sub">Desktop alerts are on</span>}
                {perm === "denied" && <span className="panel-sub">Desktop alerts are blocked in your browser settings</span>}
                {session && (
                  <button className="btn btn-ghost" onClick={() => supabase?.auth.signOut()}>
                    Sign out {session.user.email}
                  </button>
                )}
              </div>
            </div>

            {sessions.length === 0 ? (
              <div className="empty-state" style={{ fontSize: "1rem" }}>
                No call right now. When someone calls your Guardian number, the call appears here.
              </div>
            ) : (
              sessions.map((s) => (
                <GuardianCall
                  key={s.id}
                  session={s}
                  contacts={contacts.rows}
                  contactsNote={contactsNote}
                  token={session?.access_token ?? null}
                  onDismiss={() => dispatch({ type: "dismiss", id: s.id })}
                />
              ))
            )}

            <section className="panel stack">
              <div>
                <h3>Try it with a simulated call</h3>
                <p className="panel-sub" style={{ margin: "6px 0 0" }}>
                  Plays a prepared recording through the same phone-quality path as a real call. Simulated calls are
                  always labelled &quot;Simulation&quot;.
                </p>
              </div>
              <div className="verdict-actions">
                {SCENARIOS.map((s) => (
                  <button key={s.id} className="btn" disabled={simBusy !== null} onClick={() => simulate(s.id)}>
                    <Icon name="play" size={14} /> {simBusy === s.id ? "Starting…" : s.label}
                  </button>
                ))}
              </div>
              {simError && <Callout tone="error" title="Could not start the simulation">{simError}</Callout>}
            </section>

            {devMode ? (
              <Callout tone="info" title="Dev mode">
                Sign-in is not configured on this deployment, so trusted contacts, your Guardian line and call history
                are not available here.
              </Callout>
            ) : (
              <>
                <RowsEditor<TrustedContact>
                  table="trusted_contacts"
                  title="Trusted contacts"
                  intro="People you can call back on a number you already know. They are shown when Guardian asks you to verify."
                  fields={[
                    { key: "name", label: "Name", required: true },
                    { key: "relationship", label: "Relationship", placeholder: "e.g. daughter" },
                    { key: "phone", label: "Phone number", required: true, type: "tel", placeholder: "+14045550123" },
                  ]}
                  {...contacts}
                  emptyText="No trusted contacts yet."
                  render={(c) => (
                    <>
                      <strong>{c.name}</strong>
                      {c.relationship && <span style={{ color: "var(--text-dim)" }}> · {c.relationship}</span>}
                      <br />
                      <span className="mono">{c.phone}</span>
                    </>
                  )}
                />
                <RowsEditor<GuardianLine>
                  table="guardian_lines"
                  title="Guardian line"
                  intro="Your ProofVoice number forwards calls to your real phone. Give callers the ProofVoice number."
                  fields={[
                    { key: "display_name", label: "Name for this line", placeholder: "e.g. Mom's phone" },
                    { key: "twilio_number", label: "ProofVoice number", required: true, type: "tel", placeholder: "+14045550100" },
                    { key: "recipient_phone", label: "Forward calls to (your phone)", required: true, type: "tel", placeholder: "+14045550123" },
                  ]}
                  {...lines}
                  emptyText="No Guardian line set up yet."
                  render={(l) => (
                    <>
                      <strong>{l.display_name ?? "Guardian line"}</strong>
                      <br />
                      <span className="mono">
                        {l.twilio_number} → {l.recipient_phone}
                      </span>
                    </>
                  )}
                />
                {userId && <CallHistory userId={userId} refreshKey={historyKey} />}
              </>
            )}

            <p className="disclaimer">
              <Icon name="info" size={14} />
              <span>
                Calls are analysed only after the caller agrees. No audio, transcripts or voiceprints are stored. Guardian
                gives probabilistic evidence, not proof, and never ends a call on its own.
              </span>
            </p>
          </>
        )}
      </main>
      <Footer />
    </div>
  );
}
