"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import Callout from "@/components/Callout";
import Icon from "@/components/Icon";
import TimeSeriesChart from "@/components/TimeSeriesChart";
import { GenericSection } from "@/components/EvalDrawer";
import { describeError, getBenchmarks, getEngineMetrics, setEnginePolicy, startLoadTest } from "@/lib/api";
import { isFiniteNumber, num } from "@/lib/format";
import type { BenchmarkReport, ComputeAction, EngineMetrics, EnginePolicy, Scenario } from "@/lib/types";

const SPAN_S = 120;

const POLICIES: { id: EnginePolicy; desc: string }[] = [
  { id: "fifo", desc: "Every hop, served one at a time (original behaviour)." },
  { id: "dynamic", desc: "Every hop, with dynamic microbatching." },
  { id: "edf", desc: "Every hop, deadline-ordered microbatching with coalescing and load shedding." },
  { id: "adaptive", desc: "EDF plus uncertainty and voice-activity gating (Aegis)." },
];

const ACTIONS: { id: ComputeAction; label: string; color: string }[] = [
  { id: "infer", label: "infer", color: "var(--accent)" },
  { id: "escalate", label: "escalate", color: "var(--amber)" },
  { id: "screen_only", label: "screen_only (student)", color: "var(--accent)" },
  { id: "skip_silence", label: "skip_silence", color: "var(--gray)" },
  { id: "skip_clear", label: "skip_clear", color: "var(--green)" },
  { id: "shed", label: "shed", color: "var(--red)" },
];

interface Sample {
  t: number;
  p50: number | null;
  p95: number | null;
  util: number | null;
  demand: number | null;
  inferred: number | null;
}

const val = (v: number | null | undefined, unit = "", digits = 1) =>
  isFiniteNumber(v) ? `${Number.isInteger(v) ? v : v.toFixed(digits)}${unit}` : "—";

function Tile({ label, value, sub, small }: { label: string; value: string; sub?: string; small?: boolean }) {
  return (
    <div className="metric-tile">
      <div className="label">{label}</div>
      <div className="value" style={small ? { fontSize: "1rem" } : undefined}>
        {value}
      </div>
      {sub && <div className="label">{sub}</div>}
    </div>
  );
}

function Benchmarks() {
  const [reports, setReports] = useState<BenchmarkReport[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(() => {
    setError(null);
    getBenchmarks()
      .then((r) => setReports(r.reports))
      .catch((e) => setError(describeError(e)));
  }, []);
  useEffect(load, [load]);

  return (
    <section className="stack">
      <div className="panel-head" style={{ marginBottom: 0 }}>
        <h3 className="section-title">Benchmarks</h3>
        <button className="btn btn-sm" onClick={load}>
          <Icon name="refresh" size={13} /> Refresh
        </button>
      </div>
      {error && <Callout tone="error">{error}</Callout>}
      {reports && reports.length === 0 && <div className="empty-state">No benchmark reports have been generated yet.</div>}
      {reports?.map((r) => (
        <article className="panel" key={`${r.name}-${r.created}`}>
          <div className="panel-head">
            <h3>
              {r.name} <span className="badge badge-used">measured</span>
            </h3>
            <span className="panel-sub">Created {new Date(r.created).toLocaleString()}</span>
          </div>
          <GenericSection name={r.name} value={r.data} />
        </article>
      ))}
    </section>
  );
}

export default function EnginePage() {
  const [m, setM] = useState<EngineMetrics | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [hist, setHist] = useState<Sample[]>([]);
  const inFlight = useRef(false);

  useEffect(() => {
    const tick = async () => {
      if (inFlight.current) return;
      inFlight.current = true;
      try {
        const d = await getEngineMetrics();
        setM(d);
        setErr(null);
        const t = isFiniteNumber(d.t) ? d.t : Date.now() / 1000;
        setHist((h) => {
          if (h.length && h[h.length - 1].t >= t) return h;
          const next = [
            ...h,
            {
              t,
              p50: d.latency_ms?.p50 ?? null,
              p95: d.latency_ms?.p95 ?? null,
              util: d.gpu?.util_pct ?? null,
              demand: d.windows_per_s?.demand ?? null,
              inferred: d.windows_per_s?.inferred ?? null,
            },
          ];
          return next.filter((s) => s.t >= t - SPAN_S);
        });
      } catch (e) {
        setErr(describeError(e));
      } finally {
        inFlight.current = false;
      }
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  const [policyBusy, setPolicyBusy] = useState(false);
  const [policyErr, setPolicyErr] = useState<string | null>(null);
  const choosePolicy = async (p: EnginePolicy) => {
    setPolicyBusy(true);
    setPolicyErr(null);
    try {
      setM(await setEnginePolicy(p));
    } catch (e) {
      setPolicyErr(describeError(e));
    } finally {
      setPolicyBusy(false);
    }
  };

  const [lt, setLt] = useState({ calls: 10, seconds: 30, scenario: "mixed" as Scenario });
  const [ltMsg, setLtMsg] = useState<{ tone: "info" | "error"; text: string } | null>(null);
  const launch = async (e: React.FormEvent) => {
    e.preventDefault();
    setLtMsg(null);
    try {
      const r = await startLoadTest(lt);
      setLtMsg({ tone: "info", text: `Started ${r.started} simulated callers for ${lt.seconds} s.` });
    } catch (e) {
      setLtMsg({ tone: "error", text: describeError(e) });
    }
  };

  const times = hist.map((s) => s.t);
  const gpu = m?.gpu ?? null;
  const actionTotal = m ? ACTIONS.reduce((a, x) => a + (m.actions?.[x.id] ?? 0), 0) : 0;
  const batchHist = m ? Object.entries(m.batch_size?.hist ?? {}).sort((a, b) => Number(a[0]) - Number(b[0])) : [];
  const batchMax = Math.max(1, ...batchHist.map(([, v]) => v));
  const currentPolicy = POLICIES.find((p) => p.id === m?.policy);

  return (
    <div className="app-shell">
      <div className="chrome">
        <Header />
      </div>
      <main className="main-content stack" id="main" tabIndex={-1}>
        <div className="page-intro">
          <h2>Inference Control Center</h2>
          <p>
            Live measurements from the Aegis inference engine, polled every second from <span className="mono">/engine/metrics</span>.
            Charts keep the last {SPAN_S} s in this browser only.
          </p>
        </div>

        {err && (
          <Callout tone="error" title="Metrics unavailable">
            {err} Retrying every second.
          </Callout>
        )}

        <section className="panel">
          <div className="panel-head">
            <h3>Engine</h3>
            <span className="panel-sub">
              {m ? `${m.backend} · ${m.device ?? "device not reported"} · window ${m.window_s} s, hop ${m.hop_s} s, deadline ${m.deadline_ms} ms` : "Waiting for metrics…"}
            </span>
          </div>
          <div className="metric-tiles">
            <Tile label="Active calls" value={val(m?.active_calls)} sub={m ? `queue depth ${m.queue_depth}` : undefined} />
            <Tile label="Latency p50" value={val(m?.latency_ms?.p50, " ms")} sub={m?.latency_ms?.n != null ? `n = ${m.latency_ms.n}` : undefined} />
            <Tile label="Latency p95" value={val(m?.latency_ms?.p95, " ms")} />
            <Tile label="Latency p99" value={val(m?.latency_ms?.p99, " ms")} />
            <Tile label="Compute p50" value={val(m?.compute_ms?.p50, " ms")} sub={`p95 ${val(m?.compute_ms?.p95, " ms")}`} />
            <Tile label="Queue wait p50" value={val(m?.queue_wait_ms?.p50, " ms")} sub={`p95 ${val(m?.queue_wait_ms?.p95, " ms")}`} />
            {m && !gpu ? (
              <Tile label="GPU" value="NVML unavailable" sub="no GPU telemetry reported" small />
            ) : (
              <>
                <Tile label="GPU util" value={val(gpu?.util_pct, "%", 0)} sub={isFiniteNumber(gpu?.temp_c) ? `${gpu!.temp_c} °C` : undefined} />
                <Tile
                  label="VRAM used"
                  value={isFiniteNumber(gpu?.mem_used_mb) ? `${(gpu!.mem_used_mb / 1024).toFixed(1)} GB` : "—"}
                  sub={`of ${isFiniteNumber(gpu?.mem_total_mb) ? (gpu!.mem_total_mb / 1024).toFixed(1) + " GB" : "— (total not reported)"}`}
                />
                <Tile label="Power" value={val(gpu?.power_w, " W")} />
              </>
            )}
          </div>
        </section>

        <div className="grid-2">
          <section className="panel">
            <div className="panel-head">
              <h3>Latency (submit → result)</h3>
            </div>
            <TimeSeriesChart
              label="Latency percentiles"
              times={times}
              unit=" ms"
              series={[
                { label: "p50", color: "var(--accent)", values: hist.map((s) => s.p50) },
                { label: "p95", color: "var(--amber)", values: hist.map((s) => s.p95) },
              ]}
            />
          </section>
          <section className="panel">
            <div className="panel-head">
              <h3>GPU utilisation</h3>
            </div>
            {m && !gpu ? (
              <div className="empty-state">NVML unavailable: no GPU utilisation is reported.</div>
            ) : (
              <TimeSeriesChart
                label="GPU utilisation"
                times={times}
                unit="%"
                yMax={100}
                series={[{ label: "util", color: "var(--green)", values: hist.map((s) => s.util) }]}
              />
            )}
          </section>
          <section className="panel">
            <div className="panel-head">
              <h3>Windows per second</h3>
              <span className="panel-sub">hops arriving vs windows actually run (last 10 s)</span>
            </div>
            <TimeSeriesChart
              label="Demand versus inferred windows per second"
              times={times}
              series={[
                { label: "demand", color: "var(--text-faint)", values: hist.map((s) => s.demand) },
                { label: "inferred", color: "var(--accent)", values: hist.map((s) => s.inferred) },
              ]}
            />
          </section>
          <section className="panel stack">
            <div>
              <div className="panel-head" style={{ marginBottom: 8 }}>
                <h3>Actions</h3>
                <span className="panel-sub">{actionTotal.toLocaleString()} total</span>
              </div>
              <div className="bar-list">
                {ACTIONS.map((a) => {
                  const n = m?.actions?.[a.id];
                  const share = actionTotal && isFiniteNumber(n) ? n / actionTotal : 0;
                  return (
                    <div className="bar-row" key={a.id}>
                      <span className="mono">{a.label}</span>
                      <div className="bar-track">
                        <div className="bar-fill" style={{ width: `${share * 100}%`, background: a.color }} />
                      </div>
                      <span className="mono" style={{ textAlign: "right" }}>
                        {isFiniteNumber(n) ? `${n.toLocaleString()} · ${(share * 100).toFixed(0)}%` : "—"}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
            <div>
              <div className="panel-head" style={{ marginBottom: 8 }}>
                <h3>Batch size</h3>
                <span className="panel-sub">
                  mean {val(m?.batch_size?.mean, "", 2)} · max {val(m?.batch_size?.max)}
                </span>
              </div>
              {batchHist.length === 0 ? (
                <div className="empty-state">No batches recorded yet.</div>
              ) : (
                <>
                  <div className="hist-bar-row" style={{ height: 90 }} role="img" aria-label={`Batch size histogram: ${batchHist.map(([k, v]) => `${k}: ${v}`).join(", ")}`}>
                    {batchHist.map(([k, v]) => (
                      <div className="hist-bar-col" key={k} title={`batch ${k}: ${v}`}>
                        <div className="hist-bar" style={{ height: `${(v / batchMax) * 100}%`, background: "var(--accent)" }} />
                      </div>
                    ))}
                  </div>
                  <div style={{ display: "flex", gap: 2 }}>
                    {batchHist.map(([k]) => (
                      <span key={k} className="chart-axis" style={{ flex: 1, textAlign: "center" }}>
                        {k}
                      </span>
                    ))}
                  </div>
                </>
              )}
            </div>
          </section>
        </div>

        <div className="grid-2">
          <section className="panel">
            <div className="panel-head">
              <h3>Scheduling policy</h3>
              <span className="panel-sub">dev/demo switch</span>
            </div>
            <div className="segmented" role="radiogroup" aria-label="Scheduling policy">
              {POLICIES.map((p) => (
                <button
                  key={p.id}
                  role="radio"
                  aria-checked={m?.policy === p.id}
                  className="segment"
                  style={m?.policy === p.id ? { background: "var(--bg-panel-raised)", boxShadow: "0 0 0 1px var(--border-strong)" } : undefined}
                  disabled={policyBusy || !m}
                  onClick={() => choosePolicy(p.id)}
                  title={p.desc}
                >
                  {p.id}
                </button>
              ))}
            </div>
            <ul className="notes-list" style={{ marginTop: 12 }}>
              {POLICIES.map((p) => (
                <li key={p.id} style={p.id === m?.policy ? { color: "var(--text)" } : undefined}>
                  <span className="mono">{p.id}</span>: {p.desc}
                </li>
              ))}
            </ul>
            {m && !currentPolicy && <p className="policy-desc">Engine reports policy &quot;{m.policy}&quot;.</p>}
            {policyErr && <Callout tone="error">{policyErr}</Callout>}
          </section>

          <section className="panel">
            <div className="panel-head">
              <h3>Load test</h3>
              <span className="panel-sub">simulated callers through the 8 kHz phone path; engine metrics only</span>
            </div>
            <form onSubmit={launch} className="field-row" style={{ alignItems: "end" }}>
              <label className="panel-sub">
                Calls
                <br />
                <input className="text-input" style={{ minWidth: 0, width: 90 }} type="number" min={1} max={500} value={lt.calls} onChange={(e) => setLt({ ...lt, calls: Number(e.target.value) })} />
              </label>
              <label className="panel-sub">
                Seconds
                <br />
                <input className="text-input" style={{ minWidth: 0, width: 90 }} type="number" min={1} max={3600} value={lt.seconds} onChange={(e) => setLt({ ...lt, seconds: Number(e.target.value) })} />
              </label>
              <label className="panel-sub">
                Scenario
                <br />
                <select className="select-input" style={{ minHeight: 40 }} value={lt.scenario} onChange={(e) => setLt({ ...lt, scenario: e.target.value as Scenario })}>
                  <option value="mixed">mixed</option>
                  <option value="genuine">genuine</option>
                  <option value="synthetic">synthetic</option>
                </select>
              </label>
              <button className="btn btn-primary" type="submit" style={{ minHeight: 40 }}>
                <Icon name="play" size={13} /> Launch
              </button>
            </form>
            {ltMsg && (
              <div style={{ marginTop: 12 }}>
                <Callout tone={ltMsg.tone}>{ltMsg.text}</Callout>
              </div>
            )}
          </section>
        </div>

        <Benchmarks />

        {m && (
          <p className="panel-sub" style={{ margin: 0 }}>
            Last sample at {new Date(m.t * 1000).toLocaleTimeString()} · {num(m.latency_ms?.n)} requests in the latency window.
          </p>
        )}
      </main>
      <Footer />
    </div>
  );
}
