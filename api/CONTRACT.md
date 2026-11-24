# ProofVoice API contract (v1)

Single source of truth shared by backend (`api/`), frontend (`web/`) and ML (`ml/`).
All probabilities are **p_synthetic in [0,1]** (1.0 = synthetic). No field may be filled with a
made-up value: if something was not computed it is `null` / omitted and the UI says so.

Base URL: `http://127.0.0.1:8000` (see `web/.env.local`: `NEXT_PUBLIC_API_BASE_URL`, `NEXT_PUBLIC_WS_BASE_URL`).

## Status vocabulary

`insufficient_evidence` | `likely_human` | `inconclusive` | `likely_synthetic`

## AnalysisReport (returned by `POST /analyze/file`, embedded elsewhere)

```jsonc
{
  "id": "uuid",
  "file": {                       // from ffprobe / soundfile; null fields allowed
    "name": "clip.wav", "container": "wav", "codec": "pcm_s16le",
    "sample_rate": 16000, "channels": 1, "bit_depth": 16,
    "duration_s": 3.41, "bitrate": null, "metadata": {"encoder": "Lavf60"}
  },
  "synthetic_probability": 0.87,  // clip-level, calibrated for display
  "cm_score": 0.87,               // the exact value that would go into the HEARSAY TSV
  "analysis_confidence": 0.74,    // evidence sufficiency 0..1 (duration, detector agreement, SNR)
  "status": "likely_synthetic",
  "decision_threshold": 0.62,     // p_synthetic threshold selected on validation minDCF
  "detector": {"name": "…", "version": "…", "raw_score": 3.1, "latency_ms": 41.0},
  "timeline": [                   // sliding windows (default 2.0 s window, 0.5 s hop)
    {"start_ms": 0, "end_ms": 2000, "synthetic_probability": 0.8, "analysis_confidence": 0.6}
  ],
  "suspicious_regions": [
    {"start_ms": 500, "end_ms": 2500, "peak_probability": 0.93, "reason": "detector window score above threshold"}
  ],
  "time_to_confidence_ms": 1500,  // first time the rolling decision became stable & confident; null if never
  "techniques_run": ["metadata", "primary_detector", "spectral"],
  "why_run": {"spectral": "primary detector confidence weak (p=0.55)"},
  "evidence": {
    "<technique>": {
      "summary": "one plain-English sentence",
      "suspicion": 0.3,           // 0..1 branch-level suspicion or null if not a scoring branch
      "features": {"spectral_flatness_mean": 0.12},
      "flags": ["band-limited at 4 kHz"],
      "used_in_score": false      // true only if the branch feeds the cm_score
    }
  },
  "branch_log": [
    {"technique": "compression", "ran": false, "reason": "lossless PCM input", "runtime_ms": 0.0}
  ],
  "processing_ms": 210.0,
  "disclaimer": "Probabilistic evidence, not proof. …"
}
```

## Endpoints

### `GET /health`
`{"status":"ok","version":"0.1.0","detector":{"name":"…","version":"…","device":"cuda"},"grok_available":true}`

### `POST /analyze/file`
multipart form field `file` (any ffmpeg-decodable audio/video). Returns `AnalysisReport`.
Errors: 400 `{"detail":"…"}` for undecodable / empty / too long (> MAX_UPLOAD_MB).

### `WS /analyze/stream`
1. Client → text JSON: `{"type":"start","sample_rate":48000,"encoding":"f32le"|"s16le","source":"mic"|"grok"|"file"}`
2. Client → binary frames: mono PCM in the declared encoding (any chunk size, e.g. 20–100 ms).
3. Server → text JSON events:
   - `{"type":"ready"}`
   - `{"type":"analysis.window","t_ms":2500,"start_ms":500,"end_ms":2500,"synthetic_probability":0.81,
      "rolling_probability":0.77,"analysis_confidence":0.55,"status":"inconclusive",
      "time_to_confidence_ms":null,"level_dbfs":-23.1,"speech_ratio":0.8}`
   - `{"type":"analysis.final","report":AnalysisReport}` after client sends `{"type":"stop"}`
   - `{"type":"error","detail":"…"}`

### `POST /redteam/grok`
JSON `{"text": "optional line for Grok to speak", "voice": "optional"}`.
Server generates speech with xAI (key stays server-side), runs the **same** pipeline, returns
`{"source":"grok_voice"|"grok_tts"|"cached_fixture","text":"…","audio_wav_b64":"…","sample_rate":24000,"report":AnalysisReport,"note":"…"}`.
`source` must honestly say which path produced the audio.

### `WS /redteam/stream`
1. Client → `{"type":"start","text":"optional"}`
2. Server → interleaved
   - `{"type":"audio.chunk","pcm16_b64":"…","sample_rate":24000,"seq":0}` (the actual generated audio, for playback)
   - `analysis.window` events (same schema as `/analyze/stream`) computed on those same samples
   - `{"type":"redteam.done","source":"grok_voice"|"grok_tts"|"cached_fixture","report":AnalysisReport}`

### `GET /eval/summary`
Returns `outputs/eval_summary.json` verbatim (minDCF/EER/AUC tables, ablation, robustness, score histograms,
model selection). 404 if not yet generated — UI must show "not computed yet", never placeholder numbers.

### `POST /explain`
`{"report": AnalysisReport}` → `{"text":"…","model":"grok-…","grounded":true}`; the text may only restate
fields present in the report. 503 if no xAI key.

---

# v2 — Guardian (live telephone protection) and Aegis engine

All Guardian/engine endpoints are additive; v1 endpoints above are unchanged.

## Risk states (call level, never per-window)

`no_suspicious_evidence` | `inconclusive` | `suspicious` | `analysis_unavailable`

* `no_suspicious_evidence` — enough usable speech, no strong synthetic signal. **Not proof of identity.**
* `inconclusive` — too little speech, poor signal (noise, packet loss), unfamiliar channel or conflicting windows.
* `suspicious` — persistent manipulation evidence; the UI must prompt independent verification.
* `analysis_unavailable` — consent declined, overload (deadline missed), inference/network failure. The call itself continues.

`evidence` is the normalised sequential evidence S/H in `[0, 1.5]` (S = one-sided CUSUM of quality-weighted,
threshold-centred log-odds; H = alert threshold). `evidence >= 1` together with persistence ⇒ `suspicious`.

## Auth

Guardian endpoints take a Supabase access token: HTTP `Authorization: Bearer <token>`, WebSocket `?token=<token>`.
If the backend has no Supabase configuration it runs in single-user **dev mode** and accepts any/no token.

## Telephony (called by Twilio, not by the web app)

* `POST /twilio/voice` — incoming-call webhook → TwiML disclosure + `<Gather>` "press 1 to consent".
* `POST /twilio/consent` — `Digits=1` → `<Start><Stream track="inbound_track">` fork + `<Dial>` recipient;
  otherwise the call is connected **without** analysis (session `consent: "declined"`, state `analysis_unavailable`).
* `POST /twilio/status` — call status callback (finalises the session).
* `WS /twilio/media` — Twilio Media Streams (8 kHz μ-law JSON frames). Signature-checked when `TWILIO_AUTH_TOKEN` set.

## `WS /guardian/live?token=…`  (server → client only)

```jsonc
{"type":"hello","user_id":"…","dev_mode":false}
{"type":"call.started","session_id":"uuid","call_sid":"CA…","caller":"+14045550123","consent":"granted"|"declined"|"simulated",
 "started_at":"ISO-8601","scenario":null|"genuine"|"synthetic"|"mixed"}
{"type":"call.update","session_id":"uuid","t_ms":4500,"state":"inconclusive","evidence":0.42,
 "reason":"plain-English why this state",
 "window":null|{"start_ms":1500,"end_ms":4500,"synthetic_probability":0.71,"usable":true,"weight":0.8,
           "speech_s":2.4,"snr_db":24.1,"level_dbfs":-22.0,"packet_loss":0.0},
 "compute":{"policy":"adaptive","action":"infer"|"skip_silence"|"skip_clear"|"escalate"|"shed","latency_ms":31.2}}
{"type":"call.alert","session_id":"uuid","t_ms":6000,"evidence":1.12,
 "message":"Potential synthetic speech detected. Verify independently before sharing information or sending money.",
 "sms_sent":true|false}
{"type":"call.ended","session_id":"uuid","final_state":"suspicious","peak_evidence":1.31,"duration_ms":42000,
 "usable_speech_s":18.5,"windows_inferred":51,"windows_skipped":33}
```
`call.update` is sent once per hop (0.5 s of caller audio), including skipped hops (`window` = null).
Right after `hello`, the server replays `call.started` + the latest `call.update` for each of the user's active calls.
Nullable: `caller`, `scenario`, `window`, `compute.latency_ms`. A rejected token gets
`{"type":"error","detail":"sign in required"}` then close code **1008** — the UI should send the user to sign-in.

`risk_events.detail` by `kind`: `state` → `{"reason"}`, `alert` → `{"sms_sent"}`,
`unavailable` → `{"reason"}` or `{"why":"consent_declined"}`, `verification` → `{"contact_id","outcome"}`.
Per-window scores are deliberately **not** persisted (minimal storage); history shows state changes only.

## Guardian REST

* `POST /guardian/simulate` `{"scenario":"genuine"|"synthetic"|"mixed"}` (owner = the signed-in user; `dev-user` in dev mode) →
  `{"session_id":"uuid"}`. Streams a server-side sample through the **same** Twilio media path at real-time pace
  (consent `"simulated"`). For demos and development without a phone call; always labelled as simulated.
* `POST /guardian/sessions/{session_id}/verification` `{"contact_id":"uuid|null","outcome":"verified_genuine"|"could_not_reach"|"confirmed_scam"|"skipped"}`
  → `{"ok":true}` (also writable directly via Supabase RLS).
* Trusted contacts, lines, call history: read/written by the web app directly in Supabase (tables
  `guardian_lines`, `trusted_contacts`, `call_sessions`, `risk_events`; RLS = owner only). Deleting a
  `call_sessions` row cascades to its events. No audio is ever stored.

## Engine / Inference Control Center

* `GET /engine/metrics` →
```jsonc
{"policy":"adaptive","backend":"pytorch-fp16"|"tensorrt-fp16"|…,"device":"NVIDIA GeForce RTX 4060 Laptop GPU",
 "active_calls":3,"queue_depth":1,"window_s":3.0,"hop_s":0.5,"deadline_ms":1000,
 "latency_ms":{"p50":22.1,"p95":35.0,"p99":48.2,"n":1200},          // submit → result, rolling last 2000 requests
 "compute_ms":{"p50":18.0,"p95":24.0},"queue_wait_ms":{"p50":2.0,"p95":9.0},
 "windows_per_s":{"demand":6.0,"inferred":3.1},                     // hops arriving vs windows actually run (last 10 s)
 "actions":{"infer":512,"escalate":40,"skip_silence":201,"skip_clear":330,"shed":0},
 "batch_size":{"mean":2.4,"max":8,"hist":{"1":100,"2":40,"4":12}},
 "gpu":null|{"util_pct":41,"mem_used_mb":3120,"mem_total_mb":8188,"power_w":38.5,"temp_c":61},
 "t":1764012345.2}
```
  `gpu` is `null` when NVML is unavailable — never fabricated.
* `GET /metrics` — the same numbers in Prometheus text format.
* `POST /engine/policy` `{"policy":"fifo"|"dynamic"|"edf"|"adaptive"}` → metrics. Dev/demo switch.
  `fifo` = every hop, serial (original behaviour); `dynamic` = every hop, dynamic microbatching; `edf` = every hop,
  deadline-ordered microbatching + coalescing + shedding; `adaptive` = `edf` + uncertainty/VAD gating (Aegis).
* `POST /engine/loadtest` `{"calls":10,"seconds":30,"scenario":"mixed"}` → `{"started":10}`;
  spawns simulated concurrent callers in-process (not attached to any user; visible only in engine metrics).
* `GET /engine/benchmarks` → `{"reports":[{"name":"policy_comparison","created":"ISO","data":{…}}, …]}` — verbatim
  JSON from `outputs/benchmarks/*.json`; empty list if none. The UI must label which numbers are measured and when.
