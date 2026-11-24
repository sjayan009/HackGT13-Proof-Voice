# ProofVoice 2.0 — Guardian + Aegis

ProofVoice 2.0 turns the HackGT 13 deepfake detector into a real-time, consent-based phone protection product
(**Guardian**) powered by an uncertainty-aware GPU inference engine (**Aegis**). The original HEARSAY submission,
model and results are unchanged (`00_NSA_HEARSAY_SUBMISSION/`, `models/selected/model.pt`, `RESULTS.md`).

* Measured numbers: **[RESULTS_V2.md](RESULTS_V2.md)** (generated from result files)
* First real phone call: **[GUARDIAN_SETUP.md](GUARDIAN_SETUP.md)**
* Telephony research write-up: **[research/telephony_eval.md](research/telephony_eval.md)**
* API contract (v1 + v2): **[../api/CONTRACT.md](../api/CONTRACT.md)**

## What happens on a call

```
caller ─► Twilio number ─► /twilio/voice      disclosure + "press 1 to allow the check"   (signature-verified)
                           /twilio/consent ─┬─► <Dial> protected person (the conversation never waits on ProofVoice)
                                            └─► <Start><Stream> caller audio, single-use token
WS /twilio/media ─► μ-law decode (bit-exact G.711) ─► soxr 8→16 kHz ─► 8 s ring buffer (gaps zero-filled, counted as loss)
   every 0.5 s hop ─► energy VAD + SNR (call-level floor) ─► Aegis gate ─► GPU microbatch ─► channel-calibrated log-odds
   ─► risk engine (quality-weighted CUSUM, abstention, latched warning) ─► WS /guardian/live ─► dashboard + browser alert + SMS
   ─► Supabase: state changes, alerts and verification outcomes only — never audio
```

The protected person sees one of four states — *no suspicious evidence (not proof of identity)*, *inconclusive*,
*suspicious synthetic speech*, *analysis unavailable* — and, on a warning, is guided to call a **trusted contact
back on a number they already know** and record what happened. Declining consent connects the call unchecked.

## Components

| area | code |
|---|---|
| Telephony | `api/app/telephony/` — `twilio.py` (webhooks, TwiML, signatures), `media.py` (call session), `mulaw.py`, `sim.py` (Twilio-protocol simulator) |
| Signal quality | `api/app/audio/realtime.py` — VAD, SNR, call-level noise floor (shared with offline evaluation) |
| Risk engine | `api/app/risk/engine.py` — calibrated log-odds → CUSUM, quality weights, abstention reasons; `models/selected/{channel_calibration,risk_policy}.json` |
| Aegis | `engine/aegis.py` (gate + dispatch policies), `engine/backends.py` (PyTorch, TensorRT FP16), `engine/trt_build.py`, `engine/telemetry.py` (NVML, Prometheus) |
| Guardian service | `api/app/verification/` — live hub, auth (Supabase tokens), SMS, store, routes |
| Web | `web/app/guardian` (consumer), `web/app/engine` (Inference Control Center) |
| Data | Supabase project *ProofVoice Guardian*: `infrastructure/supabase/migrations/0001_guardian_schema.sql` (RLS owner-only) |
| Research | `ml/telephony_eval/` (simulated calls, channels, traces, calibration), `ml/experiments/queue.py` (training queue) |
| Benchmarks | `benchmarks/latency/backend_latency.py`, `benchmarks/load_test.py`, `benchmarks/policy_replay.py`, `benchmarks/report.py` |
| Ops | `infrastructure/Dockerfile.serve`, `infrastructure/docker-compose.yml` (API + Prometheus + Grafana), `.github/workflows/ci.yml` |

## Aegis scheduling policies

Two decisions are separated so each contribution can be measured on its own:

| policy | gate (whether a 0.5 s hop is scored) | dispatch (how scored windows run) |
|---|---|---|
| `fifo` | every hop | serial, batch 1 (original ProofVoice) |
| `dynamic` | every hop | microbatching (≤16, ≤5 ms wait) |
| `edf` | every hop | microbatching + earliest-deadline-first + per-call coalescing + shedding past deadline |
| `adaptive` | skip silent hops; score every 3rd hop on clearly genuine calls; escalate ambiguous hops | as `edf` (+ optional two-stage screening) |

## PRD §11 targets vs measured

All "measured" detection values are on **simulated** calls (`ml/telephony_eval`, speaker-disjoint eval split).

| target | MVP target | measured | status |
|---|---|---|---|
| Telephony EER on independent data | ≤ 3 % | 10.5 % window-level, simulated telephone channel | **not met**; no independent real-phone set yet |
| Genuine-call false alerts | ≤ 1 % | 0.8 % on calib calls; **4.5 % (fixed) / 2.7 % (adaptive)** on eval telephone calls; 0/20 in load tests | **not met** on held-out voices (LJ voice drives it) |
| Synthetic-call detection | ≥ 90 % | 90.8 % telephone, 94.6 % clean, 87–88 % noisy, 17 % crude replay proxy | met on clean/telephone only |
| First warning, p95 | ≤ 5 s usable speech | p50 5.0 s, **p95 8.5 s** usable speech (telephone) | **not met** at p95 |
| Per-window latency p95 under load | ≤ 150 ms | PyTorch batched: 72–95 ms @ 30 calls; TensorRT: 40–42 ms @ 40 calls | met |
| Simultaneous calls on one RTX 4060 | ≥ 20 | 40 (TensorRT, all policies), 30 (PyTorch with batching); serial FIFO collapses at 30 | met |
| GPU memory | within 8 GB | 2.8 GB peak (40 calls) | met |
| Adaptive compute reduction | ≥ 30 % | 20–29 % on telephone channels with unchanged detection (+0.5–2 s delay); 24 % in load test | **not met** |
| Controlled evaluation calls | ≥ 300 | 514 simulated calls; 0 consented human calls yet | partial |
| Persistent raw-audio retention | zero | 8 s in-memory ring buffer, dropped at call end; schema has no audio column | met by design |

What the gaps say: the detector, not the call policy, is the bottleneck. Window-level false alarms on narrowband
speech from unseen voices (20 % on telephone, up to 67 % for the LJ voice) cannot be fully fixed by persistence
or calibration (monotone calibration cannot change EER). That points to Workstream A — the controlled
codec-only and leave-one-generator-out training runs queued in `ml/experiments/queue.py` — and to collecting a
real, consented telephone evaluation set.

## Not built (by design or deferred)

* **Fraud analyst dashboard (P2)** and **developer SDK (P2)**: Shield/Engine expansions; the streaming protocol is
  documented in the contract and `app.telephony.sim` doubles as a reference client.
* **On-device inference (P3)**.
* **Learned scheduling**: the PRD says deterministic baselines first; they now exist with measurements.
* **Two-stage screening** is implemented (`AEGIS_SCREEN_MODEL_DIR`, `screen_only` action) and evaluated
  (`benchmarks/cascade_replay.py`), and is **off by default**: with the distilled 4-layer student it costs *more*
  than adaptive gating alone (1.09 vs 0.76 of fixed-rate compute on simulated telephone calls). The student
  re-runs the shared convolutional front end (it is only ~1/3 cheaper per window) and is rarely confident on
  narrowband speech. The better design is an early-exit head on the main model, so escalation reuses the
  front-end and first-layer activations instead of recomputing them.
* No automatic hang-up, blocking or accusation — by PRD §12, never on a detector score alone.

## Run it

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir api --port 8000      # set AEGIS_BACKEND=tensorrt for TRT
```
```powershell
npm --prefix web run dev                                                          # http://localhost:3000/guardian and /engine
```
```powershell
.venv\Scripts\python.exe benchmarks\load_test.py --backend tensorrt; .venv\Scripts\python.exe benchmarks\policy_replay.py; .venv\Scripts\python.exe benchmarks\report.py
```
