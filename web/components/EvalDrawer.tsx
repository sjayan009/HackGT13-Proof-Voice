"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getEvalSummary, type EvalSummaryResult } from "@/lib/api";
import type { EvalHistogram } from "@/lib/types";
import { humanizeKey, isFiniteNumber, num } from "@/lib/format";
import Callout from "./Callout";
import Icon from "./Icon";

function isHistogram(value: unknown): value is EvalHistogram {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return Array.isArray(v.bins) && Array.isArray(v.bonafide) && Array.isArray(v.spoof) && v.bins.length > 0;
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

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.length > 0 && value.every((v) => typeof v === "string");
}

function HistogramChart({ hist, title }: { hist: EvalHistogram; title: string }) {
  const maxVal = Math.max(...hist.bonafide, ...hist.spoof, 1);
  return (
    <figure style={{ margin: "0 0 20px" }}>
      <figcaption style={{ fontSize: "0.75rem", color: "var(--text-dim)", marginBottom: 8 }}>
        {humanizeKey(title)}
      </figcaption>
      <div className="hist-bar-row" role="img" aria-label={`Histogram of ${humanizeKey(title)}`}>
        {hist.bins.map((b, i) => (
          <div
            className="hist-bar-col"
            key={i}
            title={`bin ${num(b)} · bona fide ${hist.bonafide[i] ?? 0} · spoof ${hist.spoof[i] ?? 0}`}
          >
            <div
              className="hist-bar"
              style={{
                height: `${((hist.bonafide[i] ?? 0) / maxVal) * 100}%`,
                background: "var(--green)",
                opacity: 0.8,
              }}
            />
            <div
              className="hist-bar"
              style={{
                height: `${((hist.spoof[i] ?? 0) / maxVal) * 100}%`,
                background: "var(--red)",
                opacity: 0.8,
              }}
            />
          </div>
        ))}
      </div>
      <div className="chart-legend">
        <span>
          <i className="swatch" style={{ background: "var(--green)" }} /> Bona fide
        </span>
        <span>
          <i className="swatch" style={{ background: "var(--red)" }} /> Spoof
        </span>
      </div>
    </figure>
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
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c} scope="col" className={typeof rows[0]?.[c] === "number" ? "num" : undefined}>
                {humanizeKey(c)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {columns.map((c) => (
                <td key={c} className={typeof row[c] === "number" ? "num mono" : undefined}>
                  {num(row[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function KvTable({ obj }: { obj: Record<string, unknown> }) {
  return (
    <div className="table-scroll">
      <table className="data-table">
        <tbody>
          {Object.entries(obj).map(([k, v]) => (
            <tr key={k}>
              <td style={{ color: "var(--text-dim)" }}>{humanizeKey(k)}</td>
              <td className="num mono">{num(v)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function GenericSection({ name, value }: { name: string; value: unknown }) {
  if (isHistogram(value)) return <HistogramChart hist={value} title={name} />;
  if (isStringArray(value)) {
    return (
      <ul className="notes-list">
        {value.map((s, i) => (
          <li key={i}>{s}</li>
        ))}
      </ul>
    );
  }
  if (isArrayOfFlatRecords(value)) return <RecordTable rows={value} />;
  if (isFlatRecord(value)) return <KvTable obj={value} />;
  if (Array.isArray(value)) return <pre className="json">{JSON.stringify(value, null, 2)}</pre>;
  if (value && typeof value === "object") {
    return (
      <div>
        {Object.entries(value as Record<string, unknown>).map(([k, v]) => (
          <div key={k} style={{ marginBottom: 16 }}>
            {!isHistogram(v) && (
              <div style={{ fontSize: "0.8125rem", fontWeight: 600, marginBottom: 6 }}>{humanizeKey(k)}</div>
            )}
            <GenericSection name={k} value={v} />
          </div>
        ))}
      </div>
    );
  }
  return <p style={{ margin: 0, fontSize: "0.875rem" }}>{String(value)}</p>;
}

// Headline tiles, shown only for keys actually present in eval_summary.json.
const HEADLINE: { key: string; label: string; fmt: (v: number) => string }[] = [
  { key: "val_minDCF", label: "Val minDCF", fmt: (v) => v.toFixed(3) },
  { key: "EER", label: "EER", fmt: (v) => `${(v * 100).toFixed(1)}%` },
  { key: "AUC", label: "AUC", fmt: (v) => v.toFixed(3) },
  { key: "decision_threshold", label: "Threshold", fmt: (v) => v.toFixed(3) },
];

function Headline({ selected }: { selected: Record<string, unknown> }) {
  const tiles = HEADLINE.filter((h) => isFiniteNumber(selected[h.key]));
  if (tiles.length === 0) return null;
  return (
    <section className="panel">
      <div className="panel-head">
        <h3>Selected model</h3>
        {typeof selected.name === "string" && <span className="panel-sub">{selected.name}</span>}
      </div>
      <div className="metric-tiles">
        {tiles.map((t) => (
          <div className="metric-tile" key={t.key}>
            <div className="label">{t.label}</div>
            <div className="value">{t.fmt(selected[t.key] as number)}</div>
          </div>
        ))}
      </div>
      <p className="panel-sub" style={{ margin: "12px 0 0" }}>
        Official organizer minDCF (C<sub>fa</sub> = 4·C<sub>miss</sub>); lower is better.
      </p>
    </section>
  );
}

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select, textarea, [tabindex]:not([tabindex="-1"]), summary';

export default function EvalDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [mounted, setMounted] = useState(open);
  const [closing, setClosing] = useState(false);
  const [state, setState] = useState<EvalSummaryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const dialogRef = useRef<HTMLDivElement>(null);
  const returnFocus = useRef<HTMLElement | null>(null);

  const load = useCallback(() => {
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

  // Mount on open; on close, play the exit animation before unmounting.
  useEffect(() => {
    if (open) {
      returnFocus.current = document.activeElement as HTMLElement | null;
      setMounted(true);
      setClosing(false);
      return load();
    }
    if (mounted) {
      setClosing(true);
      const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      const t = setTimeout(() => {
        setMounted(false);
        setClosing(false);
        returnFocus.current?.focus?.();
      }, reduce ? 0 : 240);
      return () => clearTimeout(t);
    }
  }, [open]);

  useEffect(() => {
    if (!mounted || closing) return;
    dialogRef.current?.focus();
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      } else if (e.key === "Tab" && dialogRef.current) {
        const nodes = Array.from(dialogRef.current.querySelectorAll<HTMLElement>(FOCUSABLE));
        if (nodes.length === 0) return;
        const first = nodes[0];
        const last = nodes[nodes.length - 1];
        const active = document.activeElement;
        if (e.shiftKey && (active === first || active === dialogRef.current)) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && active === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prevOverflow;
      document.removeEventListener("keydown", onKey);
    };
  }, [mounted, closing, onClose]);

  if (!mounted) return null;

  const data = state?.kind === "ok" ? state.data : null;
  const selected =
    data && data.selected_model && typeof data.selected_model === "object"
      ? (data.selected_model as Record<string, unknown>)
      : null;

  return (
    <div className={closing ? "drawer-closing" : undefined}>
      <div className="drawer-overlay" onClick={onClose} aria-hidden />
      <div
        className="drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="eval-title"
        ref={dialogRef}
        tabIndex={-1}
      >
        <div className="drawer-head">
          <h2 id="eval-title">Evaluation</h2>
          <div className="control-group" style={{ gap: 6 }}>
            <button className="btn btn-ghost btn-icon" onClick={load} aria-label="Reload evaluation" disabled={loading}>
              <Icon name="refresh" size={16} />
            </button>
            <button className="btn btn-ghost btn-icon" onClick={onClose} aria-label="Close evaluation">
              <Icon name="close" size={18} />
            </button>
          </div>
        </div>

        <div className="drawer-body" aria-busy={loading}>
          {loading && !state && (
            <div className="stack">
              <div className="skeleton" style={{ height: 120 }} />
              <div className="skeleton" style={{ height: 200 }} />
            </div>
          )}

          {state?.kind === "not_computed" && (
            <div className="empty-state">
              Evaluation has not been computed yet. Run <code>ml/build_eval_summary.py</code> to generate it.
            </div>
          )}

          {state?.kind === "error" && (
            <Callout tone="error" title="Could not load the evaluation summary">
              {state.message}
            </Callout>
          )}

          {data && (
            <div className="appear">
              {selected && <Headline selected={selected} />}
              {Object.entries(data).map(([section, value]) => (
                <section className="panel" key={section} style={{ marginTop: 14 }}>
                  <div className="panel-head">
                    <h3 style={{ textTransform: "capitalize" }}>{humanizeKey(section)}</h3>
                  </div>
                  <GenericSection name={section} value={value} />
                </section>
              ))}
              {Object.keys(data).length === 0 && <div className="empty-state">Evaluation summary is empty.</div>}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
