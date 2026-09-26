"use client";

import { useEffect, useState } from "react";
import { getEvalSummary } from "@/lib/api";
import type { EvalHistogram, EvalSummary } from "@/lib/types";

function isHistogram(value: unknown): value is EvalHistogram {
  if (!value || typeof value !== "object") return false;
  const v = value as any;
  return (
    Array.isArray(v.bins) &&
    Array.isArray(v.bonafide) &&
    Array.isArray(v.spoof) &&
    v.bins.length > 0
  );
}

function isFlatRecord(value: unknown): value is Record<string, string | number | boolean | null> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  return Object.values(value).every(
    (v) => v === null || typeof v === "string" || typeof v === "number" || typeof v === "boolean"
  );
}

function isArrayOfFlatRecords(value: unknown): value is Record<string, unknown>[] {
  return Array.isArray(value) && value.length > 0 && value.every(isFlatRecord);
}

function HistogramChart({ hist, title }: { hist: EvalHistogram; title: string }) {
  const maxVal = Math.max(...hist.bonafide, ...hist.spoof, 1);
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ fontSize: 12, color: "var(--text-dim)", marginBottom: 8 }}>
        {title}
      </div>
      <div className="hist-bar-row">
        {hist.bins.map((b, i) => (
          <div className="hist-bar-col" key={i} title={`bin ${b}`}>
            <div
              className="hist-bar"
              style={{
                height: `${((hist.bonafide[i] ?? 0) / maxVal) * 100}%`,
                background: "var(--green)",
                opacity: 0.75,
              }}
            />
            <div
              className="hist-bar"
              style={{
                height: `${((hist.spoof[i] ?? 0) / maxVal) * 100}%`,
                background: "var(--red)",
                opacity: 0.75,
              }}
            />
          </div>
        ))}
      </div>
      <div style={{ display: "flex", gap: 14, fontSize: 11, marginTop: 6 }}>
        <span style={{ color: "var(--green)" }}>■ bonafide</span>
        <span style={{ color: "var(--red)" }}>■ spoof</span>
      </div>
    </div>
  );
}

function RecordTable({ rows }: { rows: Record<string, unknown>[] }) {
  const columns = Array.from(
    rows.reduce((set, row) => {
      Object.keys(row).forEach((k) => set.add(k));
      return set;
    }, new Set<string>())
  );
  return (
    <table className="data-table">
      <thead>
        <tr>
          {columns.map((c) => (
            <th key={c}>{c}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={i}>
            {columns.map((c) => (
              <td key={c} className="mono">
                {row[c] == null
                  ? "—"
                  : typeof row[c] === "number"
                  ? Number(row[c]).toString()
                  : String(row[c])}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function KvTable({ obj }: { obj: Record<string, unknown> }) {
  return (
    <table className="data-table">
      <tbody>
        {Object.entries(obj).map(([k, v]) => (
          <tr key={k}>
            <td>{k}</td>
            <td className="mono">
              {v == null ? "—" : typeof v === "object" ? JSON.stringify(v) : String(v)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function GenericSection({ name, value }: { name: string; value: unknown }) {
  if (isHistogram(value)) {
    return <HistogramChart hist={value} title={name} />;
  }
  if (name === "histograms" && value && typeof value === "object" && !Array.isArray(value)) {
    return (
      <div>
        {Object.entries(value as Record<string, unknown>).map(([k, v]) =>
          isHistogram(v) ? <HistogramChart key={k} hist={v} title={k} /> : (
            <div key={k} style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 12, color: "var(--text-dim)", marginBottom: 6 }}>{k}</div>
              <pre style={{ fontSize: 11, whiteSpace: "pre-wrap" }}>{JSON.stringify(v, null, 2)}</pre>
            </div>
          )
        )}
      </div>
    );
  }
  if (isArrayOfFlatRecords(value)) {
    return <RecordTable rows={value} />;
  }
  if (isFlatRecord(value)) {
    return <KvTable obj={value} />;
  }
  if (Array.isArray(value)) {
    return (
      <pre style={{ fontSize: 11, whiteSpace: "pre-wrap" }}>
        {JSON.stringify(value, null, 2)}
      </pre>
    );
  }
  if (value && typeof value === "object") {
    return (
      <div>
        {Object.entries(value as Record<string, unknown>).map(([k, v]) => (
          <div key={k} style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>{k}</div>
            <GenericSection name={k} value={v} />
          </div>
        ))}
      </div>
    );
  }
  return <div className="mono">{String(value)}</div>;
}

export default function EvalDrawer({ onClose }: { onClose: () => void }) {
  const [loading, setLoading] = useState(true);
  const [state, setState] = useState<
    | { kind: "ok"; data: EvalSummary }
    | { kind: "not_computed" }
    | { kind: "error"; message: string }
    | null
  >(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getEvalSummary().then((res) => {
      if (!cancelled) {
        setState(res);
        setLoading(false);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <>
      <div className="drawer-overlay" onClick={onClose} />
      <div className="drawer">
        <div className="drawer-head">
          <h2 style={{ margin: 0, fontSize: 16 }}>Evaluation summary</h2>
          <button className="btn" onClick={onClose}>
            Close
          </button>
        </div>

        {loading && <div className="empty-state">Loading…</div>}

        {!loading && state?.kind === "not_computed" && (
          <div className="empty-state">Evaluation not computed yet.</div>
        )}

        {!loading && state?.kind === "error" && (
          <div className="error-state">
            Could not load evaluation summary: {state.message}
          </div>
        )}

        {!loading && state?.kind === "ok" && (
          <div>
            {Object.entries(state.data).map(([section, value]) => (
              <div className="panel" key={section}>
                <h2>{section.replace(/_/g, " ")}</h2>
                <GenericSection name={section} value={value} />
              </div>
            ))}
            {Object.keys(state.data).length === 0 && (
              <div className="empty-state">Evaluation summary is empty.</div>
            )}
          </div>
        )}
      </div>
    </>
  );
}
