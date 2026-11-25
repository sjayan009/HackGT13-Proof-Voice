"use client";

// Owner data stored directly in Supabase (RLS = owner only): trusted contacts, Guardian lines,
// call history. Table/column names follow infrastructure/supabase/migrations/0001_guardian_schema.sql.

import { useCallback, useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";
import type { CallSessionRow, RiskEventRow, VerificationOutcome } from "@/lib/types";
import GuardianTimeline from "./GuardianTimeline";
import Callout from "./Callout";
import { OUTCOME_LABEL, STATE_COPY, isSimulated } from "./GuardianCall";

/** Rows of one owner table, newest first. RLS already limits them to the signed-in user. */
export function useOwnerRows<T extends { id: string | number }>(table: string, enabled: boolean, orderBy = "created_at") {
  const [rows, setRows] = useState<T[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const reload = useCallback(async () => {
    if (!supabase || !enabled) return;
    setLoading(true);
    const { data, error } = await supabase.from(table).select("*").order(orderBy, { ascending: false }).limit(200);
    setLoading(false);
    if (error) setError(error.message);
    else {
      setError(null);
      setRows((data ?? []) as T[]);
    }
  }, [table, enabled, orderBy]);

  useEffect(() => {
    reload();
  }, [reload]);

  return { rows, error, loading, reload, setError };
}

type Field = { key: string; label: string; required?: boolean; type?: string; hint?: string; placeholder?: string };

/** List + add/edit/delete form for a simple owner table (trusted_contacts, guardian_lines). */
export function RowsEditor<T extends { id: string }>({
  table,
  title,
  intro,
  fields,
  rows,
  error,
  reload,
  setError,
  render,
  emptyText,
}: {
  table: string;
  title: string;
  intro: string;
  fields: Field[];
  rows: T[];
  error: string | null;
  reload: () => Promise<void>;
  setError: (e: string | null) => void;
  render: (row: T) => React.ReactNode;
  emptyText: string;
}) {
  const blank = Object.fromEntries(fields.map((f) => [f.key, ""]));
  const [form, setForm] = useState<Record<string, string>>(blank);
  const [editing, setEditing] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!supabase) return;
    setBusy(true);
    // Empty optional fields are stored as null, never as made-up text.
    const values = Object.fromEntries(fields.map((f) => [f.key, form[f.key].trim() || null]));
    const { error } = editing
      ? await supabase.from(table).update(values).eq("id", editing)
      : await supabase.from(table).insert(values);
    setBusy(false);
    if (error) return setError(error.message);
    setForm(blank);
    setEditing(null);
    await reload();
  };

  const remove = async (row: T) => {
    if (!supabase || !window.confirm("Delete this entry?")) return;
    const { error } = await supabase.from(table).delete().eq("id", row.id);
    if (error) return setError(error.message);
    await reload();
  };

  const edit = (row: T) => {
    setEditing(row.id);
    const r = row as unknown as Record<string, unknown>;
    setForm(Object.fromEntries(fields.map((f) => [f.key, r[f.key] == null ? "" : String(r[f.key])])));
  };

  return (
    <section className="panel stack">
      <div>
        <h3>{title}</h3>
        <p className="panel-sub" style={{ margin: "6px 0 0" }}>
          {intro}
        </p>
      </div>
      {rows.length === 0 ? (
        <div className="empty-state">{emptyText}</div>
      ) : (
        <ul className="contact-list">
          {rows.map((r) => (
            <li key={r.id}>
              <span>{render(r)}</span>
              <span style={{ display: "flex", gap: 8 }}>
                <button className="btn btn-sm" onClick={() => edit(r)}>
                  Edit
                </button>
                <button className="btn btn-sm btn-danger" onClick={() => remove(r)}>
                  Delete
                </button>
              </span>
            </li>
          ))}
        </ul>
      )}
      <form onSubmit={save} className="form-grid" aria-label={editing ? `Edit ${title}` : `Add to ${title}`}>
        {fields.map((f) => (
          <div key={f.key}>
            <label htmlFor={`${table}-${f.key}`}>
              {f.label}
              {!f.required && " (optional)"}
            </label>
            <input
              id={`${table}-${f.key}`}
              className="text-input"
              type={f.type ?? "text"}
              required={f.required}
              placeholder={f.placeholder}
              value={form[f.key]}
              onChange={(e) => setForm({ ...form, [f.key]: e.target.value })}
            />
          </div>
        ))}
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn btn-primary" type="submit" disabled={busy}>
            {editing ? "Save changes" : "Add"}
          </button>
          {editing && (
            <button
              className="btn btn-ghost"
              type="button"
              onClick={() => {
                setEditing(null);
                setForm(blank);
              }}
            >
              Cancel
            </button>
          )}
        </div>
      </form>
      {error && <Callout tone="error">{error}</Callout>}
    </section>
  );
}

function fmtDate(iso: string) {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

function SessionEvidence({ sessionId }: { sessionId: string }) {
  const [events, setEvents] = useState<RiskEventRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!supabase) return;
    supabase
      .from("risk_events")
      .select("*")
      .eq("session_id", sessionId)
      .order("id", { ascending: true })
      .then(({ data, error }) => {
        if (error) setError(error.message);
        else setEvents((data ?? []) as RiskEventRow[]);
      });
  }, [sessionId]);

  if (error) return <Callout tone="error">{error}</Callout>;
  if (!events) return <div className="empty-state">Loading events…</div>;

  // History keeps sequential evidence per stored event; per-window scores are not stored.
  const points = events
    .filter((e) => e.t_ms != null && (e.kind === "state" || e.kind === "alert"))
    .map((e) => ({ t_ms: e.t_ms as number, evidence: e.evidence, p: null, usable: null, skipped: false }));
  const verifications = events.filter((e) => e.kind === "verification");

  return (
    <div className="stack">
      {points.length > 0 ? (
        <GuardianTimeline points={points} />
      ) : (
        <div className="empty-state">No evidence events were stored for this call.</div>
      )}
      {verifications.map((v) => {
        const o = v.detail?.outcome as VerificationOutcome | undefined;
        return (
          <p key={v.id} style={{ margin: 0 }}>
            Your verification: {o && OUTCOME_LABEL[o] ? OUTCOME_LABEL[o] : "recorded"} ({fmtDate(v.created_at)})
          </p>
        );
      })}
    </div>
  );
}

export function CallHistory({ userId, refreshKey }: { userId: string; refreshKey: number }) {
  const { rows, error, reload, setError } = useOwnerRows<CallSessionRow>("call_sessions", true, "started_at");
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => {
    if (refreshKey) reload();
  }, [refreshKey, reload]);

  const del = async (id: string) => {
    if (!supabase || !window.confirm("Delete this call and its events from your history? This cannot be undone.")) return;
    const { error } = await supabase.from("call_sessions").delete().eq("id", id);
    if (error) setError(error.message);
    reload();
  };

  const delAll = async () => {
    if (!supabase || !window.confirm("Delete ALL of your call history? This cannot be undone.")) return;
    const { error } = await supabase.from("call_sessions").delete().eq("user_id", userId);
    if (error) setError(error.message);
    reload();
  };

  return (
    <section className="panel">
      <div className="panel-head">
        <h3>Call history</h3>
        <button className="btn btn-danger btn-sm" onClick={delAll} disabled={rows.length === 0}>
          Delete all my call history
        </button>
      </div>
      <p className="panel-sub" style={{ margin: "0 0 8px" }}>
        No audio or transcripts are ever stored, only the time of each call and its risk events. Deleting a call removes
        its events too.
      </p>
      {rows.length === 0 && <div className="empty-state">No calls yet.</div>}
      {rows.map((s) => (
        <div key={s.id}>
          <div className="history-row">
            <div>
              <strong>{s.caller_number ?? "Unknown caller"}</strong>{" "}
              {isSimulated(s) && <span className="badge badge-warn">Simulation</span>}
              {s.alerted && <span className="badge" style={{ color: "var(--red)", background: "var(--red-bg)" }}>Alert raised</span>}
              <div className="panel-sub">
                {fmtDate(s.started_at)} ·{" "}
                {s.final_state ? STATE_COPY[s.final_state]?.title ?? s.final_state : s.ended_at ? "No final state recorded" : "In progress"}
                {s.consent === "declined" && " · caller declined the check"}
              </div>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn btn-sm" aria-expanded={open === s.id} onClick={() => setOpen(open === s.id ? null : s.id)}>
                {open === s.id ? "Hide evidence" : "View evidence"}
              </button>
              <button className="btn btn-sm btn-danger" onClick={() => del(s.id)}>
                Delete
              </button>
            </div>
          </div>
          {open === s.id && <SessionEvidence sessionId={s.id} />}
        </div>
      ))}
      {error && <Callout tone="error">{error}</Callout>}
    </section>
  );
}
