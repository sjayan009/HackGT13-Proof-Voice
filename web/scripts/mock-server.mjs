#!/usr/bin/env node
// DEV-ONLY mock of the ProofVoice backend (api/CONTRACT.md), for frontend
// development when the real backend isn't running yet.
//
// NOT imported by the app. Run manually: `npm run mock` (listens on :8000).
//
// Implements: GET /health, POST /analyze/file, WS /analyze/stream,
// POST /redteam/grok, WS /redteam/stream, GET /eval/summary, POST /explain.
// v2: WS /guardian/live, POST /guardian/simulate, POST /guardian/sessions/{id}/verification,
// GET/POST /engine/metrics, /engine/policy, /engine/loadtest, /engine/benchmarks.
// MOCK_GPU=null makes /engine/metrics report gpu: null (the "NVML unavailable" path).
// All numbers are synthetic/random — never treat this as real detection output.

import http from "node:http";
import crypto from "node:crypto";
import { WebSocketServer } from "ws";

const PORT = process.env.MOCK_PORT || 8000;

function randStatus(p) {
  if (p < 0.35) return "likely_human";
  if (p < 0.5) return "inconclusive";
  if (p < 0.65) return "inconclusive";
  return "likely_synthetic";
}

function fakeReport(name = "uploaded.wav", durationS = 6.0) {
  const p = Math.random();
  const status = randStatus(p);
  const windowMs = 2000;
  const hopMs = 500;
  const timeline = [];
  for (let start = 0; start + windowMs <= durationS * 1000; start += hopMs) {
    timeline.push({
      start_ms: start,
      end_ms: start + windowMs,
      synthetic_probability: Math.max(0, Math.min(1, p + (Math.random() - 0.5) * 0.3)),
      analysis_confidence: 0.5 + Math.random() * 0.4,
    });
  }
  if (timeline.length === 0) {
    timeline.push({
      start_ms: 0,
      end_ms: durationS * 1000,
      synthetic_probability: p,
      analysis_confidence: 0.6,
    });
  }
  const suspicious_regions =
    p > 0.5
      ? [
          {
            start_ms: Math.floor(durationS * 200),
            end_ms: Math.floor(durationS * 500),
            peak_probability: Math.min(1, p + 0.1),
            reason: "detector window score above threshold",
          },
        ]
      : [];

  return {
    id: crypto.randomUUID(),
    file: {
      name,
      container: "wav",
      codec: "pcm_s16le",
      sample_rate: 16000,
      channels: 1,
      bit_depth: 16,
      duration_s: durationS,
      bitrate: null,
      metadata: { encoder: "mock" },
    },
    synthetic_probability: p,
    cm_score: p,
    analysis_confidence: 0.5 + Math.random() * 0.4,
    status,
    decision_threshold: 0.62,
    detector: { name: "mock-detector", version: "0.0.0-mock", raw_score: p * 4, latency_ms: 12.3 },
    timeline,
    suspicious_regions,
    time_to_confidence_ms: p > 0.5 ? 1800 : null,
    techniques_run: ["metadata", "primary_detector", "spectral"],
    why_run: p > 0.4 && p < 0.7 ? { spectral: "primary detector confidence weak" } : {},
    evidence: {
      metadata: {
        summary: "File metadata is consistent with a standard PCM recording.",
        suspicion: null,
        features: { container_ok: 1, header_valid: 1 },
        flags: [],
        used_in_score: false,
      },
      primary_detector: {
        summary: `Primary detector estimates p(synthetic) = ${p.toFixed(2)}.`,
        suspicion: p,
        features: { raw_score: p * 4, calibration_temp: 1.0 },
        flags: p > 0.6 ? ["score above decision threshold"] : [],
        used_in_score: true,
      },
      spectral: {
        summary: "Spectral flatness within expected range for natural speech.",
        suspicion: Math.max(0, p - 0.1),
        features: { spectral_flatness_mean: 0.12, band_limit_hz: 8000 },
        flags: [],
        used_in_score: false,
      },
    },
    branch_log: [
      { technique: "metadata", ran: true, reason: "always runs", runtime_ms: 0.4 },
      { technique: "primary_detector", ran: true, reason: "always runs", runtime_ms: 41.2 },
      {
        technique: "spectral",
        ran: true,
        reason: "supporting evidence",
        runtime_ms: 8.1,
      },
      {
        technique: "compression",
        ran: false,
        reason: "lossless PCM input",
        runtime_ms: 0.0,
      },
    ],
    processing_ms: 62.5,
    disclaimer:
      "Probabilistic evidence, not proof. ProofVoice estimates the likelihood of synthetic speech; it cannot guarantee ground truth.",
  };
}

const server = http.createServer(async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "*");
  res.setHeader("Access-Control-Allow-Methods", "*");
  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  if (req.method === "GET" && req.url === "/health") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(
      JSON.stringify({
        status: "ok",
        version: "0.0.0-mock",
        detector: { name: "mock-detector", version: "0.0.0-mock", device: "cpu" },
        grok_available: true,
      })
    );
    return;
  }

  if (req.method === "POST" && req.url === "/analyze/file") {
    // Don't bother parsing multipart; just fabricate a report after a short delay.
    let body = [];
    req.on("data", (c) => body.push(c));
    req.on("end", () => {
      setTimeout(() => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(fakeReport("uploaded-file.wav", 3 + Math.random() * 8)));
      }, 400);
    });
    return;
  }

  if (req.method === "POST" && req.url === "/redteam/grok") {
    let body = [];
    req.on("data", (c) => body.push(c));
    req.on("end", () => {
      const report = fakeReport("grok-fallback.wav", 4.0);
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          source: "cached_fixture",
          text: "This is a mock red-team utterance.",
          audio_wav_b64: "",
          sample_rate: 24000,
          report,
          note: "mock server: no real Grok call made",
        })
      );
    });
    return;
  }

  if (req.method === "POST" && req.url === "/explain") {
    let body = [];
    req.on("data", (c) => body.push(c));
    req.on("end", () => {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          text: "Mock explanation grounded only in the submitted report fields.",
          model: "mock-explainer",
          grounded: true,
        })
      );
    });
    return;
  }

  if (req.method === "GET" && req.url === "/eval/summary") {
    // Toggle to 404 by setting MOCK_EVAL=404 to test the "not computed" UI path.
    if (process.env.MOCK_EVAL === "404") {
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ detail: "not computed" }));
      return;
    }
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(
      JSON.stringify({
        metrics: { minDCF: 0.041, EER: 0.032, AUC: 0.985 },
        ablation: [
          { technique: "primary_detector_only", minDCF: 0.061 },
          { technique: "primary_detector+spectral", minDCF: 0.041 },
        ],
        histograms: {
          score_distribution: {
            bins: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            bonafide: [120, 90, 40, 20, 10, 5, 2, 1, 0, 0],
            spoof: [2, 4, 6, 10, 20, 30, 60, 90, 110, 130],
          },
        },
      })
    );
    return;
  }

  if (await handleV2(req, res)) return;

  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ detail: "not found (mock server)" }));
});

// ------------------------------------------------------------------
// v2 mock: Guardian + engine. Fake numbers for UI development only.
// ------------------------------------------------------------------

const guardianClients = new Set();
const broadcast = (evt) => {
  const msg = JSON.stringify(evt);
  for (const ws of guardianClients) if (ws.readyState === 1) ws.send(msg);
};

function readJson(req) {
  return new Promise((resolve) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch {
        resolve({});
      }
    });
  });
}

function sendJson(res, status, obj) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(obj));
  return true;
}

const engine = {
  policy: "adaptive",
  actions: { infer: 0, escalate: 0, skip_silence: 0, skip_clear: 0, shed: 0 },
  batchHist: {},
  liveCalls: 0, // simulated guardian calls
  loadCalls: [], // [{ until }]
  lat: 22,
};
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const gauss = () => (Math.random() + Math.random() + Math.random() - 1.5) / 1.5;

function activeCalls() {
  const now = Date.now();
  engine.loadCalls = engine.loadCalls.filter((c) => c.until > now);
  return engine.liveCalls + engine.loadCalls.length;
}

function countAction(a) {
  engine.actions[a] = (engine.actions[a] || 0) + 1;
  if (a === "infer" || a === "escalate") {
    const b = String(1 + Math.floor(Math.random() * Math.min(8, 1 + activeCalls())));
    engine.batchHist[b] = (engine.batchHist[b] || 0) + 1;
  }
}

// Load-test callers each produce one hop per 0.5 s.
setInterval(() => {
  const n = engine.loadCalls.filter((c) => c.until > Date.now()).length;
  for (let i = 0; i < n; i++) {
    const r = Math.random();
    countAction(r < 0.45 ? "infer" : r < 0.5 ? "escalate" : r < 0.75 ? "skip_silence" : "skip_clear");
  }
}, 500);

function engineMetrics() {
  const calls = activeCalls();
  engine.lat = clamp(engine.lat + gauss() * 2 + (18 + calls * 1.5 - engine.lat) * 0.2, 8, 400);
  const p50 = engine.lat;
  const demand = calls * 2;
  const total = Object.values(engine.actions).reduce((a, b) => a + b, 0);
  let sum = 0, cnt = 0, max = 0;
  for (const [k, v] of Object.entries(engine.batchHist)) {
    sum += Number(k) * v;
    cnt += v;
    max = Math.max(max, Number(k));
  }
  return {
    policy: engine.policy,
    backend: "mock-backend",
    device: "MOCK DEVICE (synthetic data)",
    active_calls: calls,
    queue_depth: Math.max(0, Math.round(calls / 4 + gauss())),
    window_s: 3.0,
    hop_s: 0.5,
    deadline_ms: 1000,
    latency_ms: { p50, p95: p50 * 1.6, p99: p50 * 2.2, n: Math.min(2000, total) },
    compute_ms: { p50: p50 * 0.8, p95: p50 * 1.1 },
    queue_wait_ms: { p50: p50 * 0.1, p95: p50 * 0.4 },
    windows_per_s: { demand, inferred: engine.policy === "adaptive" ? demand * 0.55 : demand },
    actions: engine.actions,
    batch_size: { mean: cnt ? sum / cnt : null, max: cnt ? max : null, hist: engine.batchHist },
    gpu:
      process.env.MOCK_GPU === "null"
        ? null
        : {
            util_pct: Math.round(clamp(5 + calls * 6 + gauss() * 4, 0, 100)),
            mem_used_mb: 2900 + calls * 20,
            mem_total_mb: 8188,
            power_w: Number((12 + calls * 3 + gauss()).toFixed(1)),
            temp_c: 50 + Math.min(30, calls),
          },
    t: Date.now() / 1000,
  };
}

const logit = (p) => {
  const q = clamp(p, 1e-3, 1 - 1e-3);
  return Math.log(q / (1 - q));
};

function runSimulation(scenario) {
  const session_id = crypto.randomUUID();
  const hops = 60; // 30 s of caller audio
  let S = 0, usableS = 0, inferred = 0, skipped = 0, peak = 0, over = 0, hop = 0;
  let suspicious = false;
  engine.liveCalls++;
  broadcast({
    type: "call.started",
    session_id,
    call_sid: "CAmock" + session_id.slice(0, 8),
    caller: "+14045550123",
    consent: "simulated",
    started_at: new Date().toISOString(),
    scenario,
  });
  const timer = setInterval(() => {
    hop++;
    const t_ms = hop * 500;
    const synthNow = scenario === "synthetic" || (scenario === "mixed" && t_ms > 12000);
    const r = Math.random();
    let window = null;
    let action;
    if (hop < 4 || r < 0.15) action = "skip_silence";
    else if (r < 0.3 && usableS > 4 && !synthNow) action = "skip_clear";
    else {
      action = r > 0.92 ? "escalate" : "infer";
      const p = clamp((synthNow ? 0.85 : 0.15) + gauss() * 0.12, 0, 1);
      const usable = Math.random() > 0.12;
      const weight = usable ? 0.6 + Math.random() * 0.4 : 0;
      if (usable) usableS += 0.5;
      S = Math.max(0, S + weight * (logit(p) - logit(0.5)) * 0.06);
      window = {
        start_ms: Math.max(0, t_ms - 3000),
        end_ms: t_ms,
        synthetic_probability: p,
        usable,
        weight,
        speech_s: Number((1.5 + Math.random() * 1.4).toFixed(1)),
        snr_db: Number((usable ? 18 + Math.random() * 10 : 4 + Math.random() * 4).toFixed(1)),
        level_dbfs: Number((-30 + Math.random() * 14).toFixed(1)),
        packet_loss: usable ? 0 : 0.08,
      };
    }
    countAction(action);
    if (window) inferred++;
    else skipped++;
    const evidence = Math.min(1.5, S);
    peak = Math.max(peak, evidence);
    over = evidence >= 1 ? over + 1 : 0;
    let state, reason;
    if (suspicious || over >= 2) {
      state = "suspicious";
      reason = "Synthetic-speech evidence stayed above the alert line across several stretches of speech.";
    } else if (usableS < 3) {
      state = "inconclusive";
      reason = "Not enough clear speech from the caller yet.";
    } else if (evidence < 0.5) {
      state = "no_suspicious_evidence";
      reason = "Enough clear speech, and no strong synthetic signal so far.";
    } else {
      state = "inconclusive";
      reason = "Some windows look unusual, but the evidence is not persistent yet.";
    }
    broadcast({
      type: "call.update",
      session_id,
      t_ms,
      state,
      evidence,
      reason,
      window,
      compute: { policy: engine.policy, action, latency_ms: window ? Number((engine.lat + gauss() * 3).toFixed(1)) : null },
    });
    if (state === "suspicious" && !suspicious) {
      suspicious = true;
      broadcast({
        type: "call.alert",
        session_id,
        t_ms,
        evidence,
        message: "Potential synthetic speech detected. Verify independently before sharing information or sending money.",
        sms_sent: false,
      });
    }
    if (hop >= hops) {
      clearInterval(timer);
      engine.liveCalls--;
      broadcast({
        type: "call.ended",
        session_id,
        final_state: state,
        peak_evidence: peak,
        duration_ms: t_ms,
        usable_speech_s: usableS,
        windows_inferred: inferred,
        windows_skipped: skipped,
      });
    }
  }, 500);
  return session_id;
}

async function handleV2(req, res) {
  const path = new URL(req.url, "http://x").pathname;
  if (req.method === "POST" && path === "/guardian/simulate") {
    const { scenario } = await readJson(req);
    if (!["genuine", "synthetic", "mixed"].includes(scenario)) return sendJson(res, 422, { detail: "bad scenario" });
    return sendJson(res, 200, { session_id: runSimulation(scenario) });
  }
  const v = path.match(/^\/guardian\/sessions\/([^/]+)\/verification$/);
  if (req.method === "POST" && v) {
    console.log("[mock] verification", v[1], await readJson(req));
    return sendJson(res, 200, { ok: true });
  }
  if (req.method === "GET" && path === "/engine/metrics") return sendJson(res, 200, engineMetrics());
  if (req.method === "POST" && path === "/engine/policy") {
    const { policy } = await readJson(req);
    if (!["fifo", "dynamic", "edf", "adaptive"].includes(policy)) return sendJson(res, 422, { detail: "bad policy" });
    engine.policy = policy;
    return sendJson(res, 200, engineMetrics());
  }
  if (req.method === "POST" && path === "/engine/loadtest") {
    const { calls = 10, seconds = 30 } = await readJson(req);
    const until = Date.now() + seconds * 1000;
    for (let i = 0; i < calls; i++) engine.loadCalls.push({ until });
    return sendJson(res, 200, { started: calls });
  }
  if (req.method === "GET" && path === "/engine/benchmarks") {
    return sendJson(res, 200, {
      reports: [
        {
          name: "mock_policy_comparison",
          created: new Date(Date.now() - 3600_000).toISOString(),
          data: {
            note: "MOCK SERVER: synthetic numbers for UI development, not measurements",
            policies: [
              { policy: "fifo", calls: 16, p95_ms: 410.2, deadline_miss_rate: 0.12 },
              { policy: "dynamic", calls: 16, p95_ms: 120.5, deadline_miss_rate: 0.01 },
              { policy: "edf", calls: 16, p95_ms: 88.1, deadline_miss_rate: 0.0 },
              { policy: "adaptive", calls: 16, p95_ms: 41.7, deadline_miss_rate: 0.0 },
            ],
          },
        },
      ],
    });
  }
  return false;
}

const wss = new WebSocketServer({ noServer: true });

server.on("upgrade", (req, socket, head) => {
  if (new URL(req.url, "http://x").pathname === "/guardian/live") {
    wss.handleUpgrade(req, socket, head, (ws) => {
      guardianClients.add(ws);
      ws.send(JSON.stringify({ type: "hello", user_id: "mock-user", dev_mode: true }));
      ws.on("close", () => guardianClients.delete(ws));
    });
  } else if (req.url === "/analyze/stream") {
    wss.handleUpgrade(req, socket, head, (ws) => handleAnalyzeStream(ws));
  } else if (req.url === "/redteam/stream") {
    wss.handleUpgrade(req, socket, head, (ws) => handleRedteamStream(ws));
  } else {
    socket.destroy();
  }
});

function handleAnalyzeStream(ws) {
  let started = false;
  let tMs = 0;
  let timer = null;
  let rolling = 0.3 + Math.random() * 0.2;

  ws.on("message", (msg, isBinary) => {
    if (!isBinary) {
      try {
        const data = JSON.parse(msg.toString());
        if (data.type === "start" && !started) {
          started = true;
          ws.send(JSON.stringify({ type: "ready" }));
          timer = setInterval(() => {
            tMs += 500;
            rolling = Math.max(0, Math.min(1, rolling + (Math.random() - 0.5) * 0.08));
            ws.send(
              JSON.stringify({
                type: "analysis.window",
                t_ms: tMs,
                start_ms: Math.max(0, tMs - 2000),
                end_ms: tMs,
                synthetic_probability: Math.max(0, Math.min(1, rolling + (Math.random() - 0.5) * 0.1)),
                rolling_probability: rolling,
                analysis_confidence: Math.min(1, tMs / 6000),
                status: randStatus(rolling),
                time_to_confidence_ms: tMs > 2500 ? tMs : null,
                level_dbfs: -20 + Math.random() * 10,
                speech_ratio: 0.7 + Math.random() * 0.2,
              })
            );
          }, 500);
        } else if (data.type === "stop") {
          if (timer) clearInterval(timer);
          ws.send(
            JSON.stringify({
              type: "analysis.final",
              report: fakeReport("live-session.wav", tMs / 1000 || 4),
            })
          );
          ws.close();
        }
      } catch {
        // ignore
      }
    }
    // binary PCM frames are ignored by the mock — it just fabricates windows on a timer
  });

  ws.on("close", () => {
    if (timer) clearInterval(timer);
  });
}

function handleRedteamStream(ws) {
  ws.on("message", (msg, isBinary) => {
    if (isBinary) return;
    try {
      const data = JSON.parse(msg.toString());
      if (data.type === "start") {
        let seq = 0;
        const chunkTimer = setInterval(() => {
          const samples = new Int16Array(2400);
          for (let i = 0; i < samples.length; i++) {
            samples[i] = Math.round(Math.sin(i / 10) * 3000);
          }
          const b64 = Buffer.from(samples.buffer).toString("base64");
          ws.send(
            JSON.stringify({
              type: "audio.chunk",
              pcm16_b64: b64,
              sample_rate: 24000,
              seq: seq++,
            })
          );
          ws.send(
            JSON.stringify({
              type: "analysis.window",
              t_ms: seq * 100,
              start_ms: Math.max(0, seq * 100 - 2000),
              end_ms: seq * 100,
              synthetic_probability: 0.7 + Math.random() * 0.2,
              rolling_probability: 0.7 + Math.random() * 0.1,
              analysis_confidence: Math.min(1, seq / 20),
              status: "likely_synthetic",
              time_to_confidence_ms: seq > 10 ? seq * 100 : null,
              level_dbfs: -15,
              speech_ratio: 0.9,
            })
          );
          if (seq >= 15) {
            clearInterval(chunkTimer);
            ws.send(
              JSON.stringify({
                type: "redteam.done",
                source: "cached_fixture",
                report: fakeReport("grok-stream.wav", 3.0),
              })
            );
            ws.close();
          }
        }, 120);
      }
    } catch {
      // ignore
    }
  });
}

server.listen(PORT, () => {
  console.log(`[mock] ProofVoice mock backend listening on http://127.0.0.1:${PORT}`);
  console.log("[mock] This is DEV-ONLY fake data. Never used by the real app build.");
});
