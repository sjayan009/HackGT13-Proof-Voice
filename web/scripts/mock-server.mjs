#!/usr/bin/env node
// DEV-ONLY mock of the ProofVoice backend (api/CONTRACT.md), for frontend
// development when the real backend isn't running yet.
//
// NOT imported by the app. Run manually: `npm run mock` (listens on :8000).
//
// Implements: GET /health, POST /analyze/file, WS /analyze/stream,
// POST /redteam/grok, WS /redteam/stream, GET /eval/summary, POST /explain.
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

  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ detail: "not found (mock server)" }));
});

const wss = new WebSocketServer({ noServer: true });

server.on("upgrade", (req, socket, head) => {
  if (req.url === "/analyze/stream") {
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
