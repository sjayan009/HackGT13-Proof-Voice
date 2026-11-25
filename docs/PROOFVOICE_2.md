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
| Telephony EER on independent data | ≤ 3 % | 10.5 % window-level, simulated telephone channel (best experimental run 7.4 %, not promoted) | **not met**; no independent real-phone set yet |
| Genuine-call false alerts | ≤ 1 % | 0.8 % on calib calls; **4.5 % (fixed) / 2.7 % (adaptive)** on eval telephone calls; 0/20 in load tests | **not met** on held-out voices (LJ voice drives it) |
| Synthetic-call detection | ≥ 90 % | 90.8 % telephone, 94.6 % clean, 87–88 % noisy, 17 % crude replay proxy | met on clean/telephone only |
| First warning, p95 | ≤ 5 s usable speech | p50 5.0 s, **p95 8.5 s** usable speech (telephone) | **not met** at p95 |
| Per-window latency p95 under load | ≤ 150 ms | PyTorch batched: 72–95 ms @ 30 calls; TensorRT: 40–42 ms @ 40 calls | met |
| Simultaneous calls on one RTX 4060 | ≥ 20 | 40 (TensorRT, all policies), 30 (PyTorch with batching); serial FIFO collapses at 30 | met |
| GPU memory | within 8 GB | 2.8 GB peak (40 calls) | met |
| Adaptive compute reduction | ≥ 30 % | 20–29 % on telephone channels with unchanged detection (+0.5–2 s delay); 24 % in load test | **not met** |
| Controlled evaluation calls | ≥ 300 | 514 simulated calls; 2 real consented phone calls (see below) | partial |
| Persistent raw-audio retention | zero | 8 s in-memory ring buffer, dropped at call end; schema has no audio column | met by design |

What the gaps say: the detector, not the call policy, is the bottleneck. Window-level false alarms on narrowband
speech from unseen voices (20 % on telephone, up to 67 % for the LJ voice) cannot be fully fixed by persistence
or calibration (monotone calibration cannot change EER). Workstream A's five controlled training runs
(`ml/experiments/queue.py`, results in RESULTS_V2 → *Training experiments* and `MODEL_STRATEGY.md`) found no
replacement for v3: codec-only augmentation lost on clean validation in both seeds, and its small phone-EER gain
is within run-to-run spread. One lead: training **without PlayHT** gave 7.4 % phone EER, 0.047 pooled minDCF and
better ElevenLabs detection than v3, but regressed on 20 dB noise and is a single seed — a second seed comes
before any promotion. A real, consented telephone evaluation set is still missing.

## First real phone calls (Nov 25, 2026)

Two consented self-test calls over the public telephone network into a Twilio number, through the full
production path: signed webhooks, consent prompt, Media Streams over a Cloudflare tunnel, Aegis on the RTX 4060,
the risk engine, the live dashboard, and Supabase (call times and state changes only, no audio).

| call | what was on the line | final state | notes |
|---|---|---|---|
| 1 (39 s) | the protected person talking normally | no suspicious evidence | flickered to *inconclusive* twice for 1–2 s, then settled |
| 2 (78 s) | a synthetic voice played from a laptop speaker into the phone | **suspicious** | warning 73 s after the call started; this includes the consent menu and the time to start playback |

The same session also ran three simulated synthetic calls from the dashboard (all *suspicious* within 9–11 s), and an
offline batch of 30 simulated genuine calls through the same media path flagged 0/30. Two calls from one speaker
show the system works end to end on real telephone audio; they are not an accuracy estimate. The calls used
**self-test mode** (the protected phone calls its own line, so nothing is dialed), which exists for one-phone setups.

## Voice-message checker (public site)

`web/app/check` is a consumer page: upload a voice message (WhatsApp/Telegram Opus, iPhone M4A, MP3, voicemail…)
and get a plain-language verdict, the seconds that looked synthetic, and what to do before sending money. It uses
`POST /analyze/file`; audio is deleted after the request. Verified with Opus, AAC and MP3 encodings of real and
synthetic speech (correct in every case; a real voice as MP3 came back *can't tell*, the cautious answer).

Hosting:

* **Web:** Vercel project `proofvoice-check` (https://proofvoice-check.vercel.app), built with
  `NEXT_PUBLIC_SITE_MODE=public` so only `/check` is shown and the other pages redirect to it.
* **Backend:** `deploy/modal_app.py` on Modal with `PROOFVOICE_PUBLIC=1`: only `/health` and `/analyze/file`,
  Guardian/telephony/engine off, 10 checks per IP per 10 min, 20 MB / 60 s limits, one container at most,
  scales to zero after 2 min. On 8 CPUs a 14 s voice note takes 5–7 s (about 17 s from cold); `GPU = "T4"` in
  the file makes it ~1–2 s once the Modal account has a payment method.
* **Status:** the Modal app is **stopped** (Nov 25, 2026), so the site loads but cannot check messages. Restart with
  `.venv/Scripts/modal deploy deploy/modal_app.py`.

Not yet done before wider use: an accuracy benchmark on real voice messages and current-generation AI voices, a
privacy/accuracy page, and a custom domain.



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
