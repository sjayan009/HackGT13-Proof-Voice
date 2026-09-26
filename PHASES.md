# PHASES.md — ProofVoice Execution Plan

## Operating rule

Every phase must end in a working artifact.

Order:

**official metric → baseline → better model → forensic breadth → live system → Grok → polish → freeze**

---

## Phase 0 — Official package setup

### Tasks

Download / locate:

- `DiffSSD.zip`
- `LJRealResampled.zip`
- `HackGTHearsayTesting.zip`
- `HearsayScoreKey4TeamX.tsv`
- `HackGTMinDCF.zip`
- official instructions

Extract:
- DiffSSD
- LJ real corpus
- scoring package

Leave the held-out test archive immutable.

Audit:
- hardware
- CUDA
- ffmpeg
- Python
- Node
- disk space

### Exit gate

- all official paths known
- training corpora readable
- scoring package executable
- test archive count = 1,671
- template count = 1,671

---

## Phase 1 — Reproduce official minDCF

### Goal

Make scoring trustworthy before training models.

### Tasks

- inspect `HackGTMinDCF.zip`
- identify official invocation
- wrap it in project code
- write parity tests
- document false-alarm / miss assumptions

### Exit gate

`python ml/evaluate_mindcf.py` reproduces organizer scoring behavior on a fixture.

---

## Phase 2 — Dataset audit

### Tasks

For LJReal and DiffSSD:

- count files
- inspect sample rates
- inspect durations
- inspect labels
- inspect generator metadata
- inspect speaker metadata
- inspect class imbalance
- estimate training-time cost

Create:
- `outputs/dataset_audit.json`
- `RESULTS.md` section

### Split strategy

If generator IDs exist:
- grouped generator split
- unseen-generator validation

Also:
- stratified iteration split

### Exit gate

Training/validation manifests are frozen.

---

## Phase 3 — Baseline detector

### Goal

Get a real minDCF quickly.

### Tasks

Benchmark one strong pretrained detector.

Required metrics:
- minDCF
- AUC
- EER
- FPR/FNR
- inference latency

Generate first syntactically valid held-out TSV.

### Exit gate

- baseline score exists
- official TSV validator passes
- model reloads reproducibly

---

## Phase 4 — Model benchmark race

### Goal

Find the best practical competition detector.

Benchmark 2–4 feasible model families.

For each record:
- minDCF
- AUC
- EER
- clean latency
- memory
- implementation risk

Potential families:
- AASIST
- RawNet
- WavLM/wav2vec-based detector
- ASVspoof baseline

### Exit gate

Create:

`outputs/model_selection.md`

Choose the winner using minDCF first.

---

## Phase 5 — Forensic breadth

### Goal

Earn the 20% forensic-depth points without destroying performance.

Implement:

- metadata/container analyzer
- spectral/prosody branch
- compression/transcoding branch
- splice/discontinuity branch
- optional speaker consistency

For every branch:
- measure standalone signal usefulness
- measure fusion effect
- log explanation value

### Exit gate

Ablation table exists.

Delete branches that are useless and add no explainability.

---

## Phase 6 — Agentic orchestration

### Goal

Make branch selection intelligent.

Rules should use:
- codec
- file quality
- primary detector confidence
- local anomalies
- analysis cost

Do not brute-force every analysis on every file.

### Exit gate

Each inference report includes:
- branches run
- why they ran
- runtime contribution

---

## Phase 7 — Robustness

### Minimum transforms

- MP3
- Opus
- noise
- gain
- repeated transcoding

Optional:
- telephony
- replay
- room response
- clipping

Measure:
- minDCF delta
- FPR delta
- score drift

### Exit gate

Robustness table exists in `RESULTS.md`.

---

## Phase 8 — Temporal Trust Timeline

### Tasks

- rolling window scoring
- suspicious-region merging
- time-to-confidence
- uncertainty / evidence state
- clip-level aggregation

### Exit gate

A file analysis returns:
- file score
- timeline
- regions
- evidence
- branch log

---

## Phase 9 — Live mode

### Tasks

- microphone capture
- WebSocket
- rolling buffer
- live inference
- stable UI updates
- reconnect behavior

### Exit gate

Human speech produces real rolling scores in browser.

---

## Phase 10 — Grok red-team

### Tasks

- connect to Grok Voice
- capture actual synthetic audio
- send same bytes into ProofVoice
- show score evolution
- calculate time-to-confidence
- build fallback using xAI streaming TTS / cached fixture

### Exit gate

One-click Grok red-team demo works.

---

## Phase 11 — Judge UX

### Screens

- Forensic Lab
- Live Trust
- Red Team
- Evaluation drawer

### Demo goal

A judge understands the system in <10 seconds.

### Exit gate

No mock results remain.

---

## Phase 12 — Draft TSV for NSA review

Do not use the one-time review on the naive baseline.

Use it after:
- model selected
- fusion decision made
- validation complete
- TSV validator passes

Prepare:
- team prediction TSV
- config hash
- validation summary

Then stop and ask the user before submitting for the one-time NSA review.

---

## Phase 13 — Docker

### Required

- reproducible build
- offline HEARSAY inference
- model loading
- exact TSV generation command
- no secret dependency

### Exit gate

Fresh container generates predictions successfully.

---

## Phase 14 — Reliability freeze

No new features.

Test:
- fresh clone
- clean Docker build
- no secret in git
- test count
- TSV exactness
- offline demo fixtures
- Chrome
- venue Wi-Fi failure
- Grok fallback
- final held-out inference

Tag a final working commit.

---

## Phase 15 — Submission

### NSA

- source repo
- Docker
- prediction TSV
- methodology
- forensic breadth
- documentation

### Meta

- working prototype
- 2–3 minute video
- public repo
- human-connection writeup

### SpaceXAI

Show:
- Cursor used
- Grok Voice red-team path
- meaningful societal problem

### Oracle

Show:
- minDCF
- ablation
- robustness
- temporal visualization
- live scoring

---

## Demo script

### 0–15s
"Three seconds of speech can be enough to clone a voice. Our question is: how quickly can we know the audio is synthetic?"

### 15–35s
Human sample → low synthetic probability.

### 35–60s
Grok Voice synthetic sample → same detector → probability rises.

### 60–80s
Open forensic evidence and suspicious timeline.

### 80–95s
Show robustness / ablation table.

### 95–110s
Show exact NSA TSV pipeline and Docker reproducibility.

### Closing line

> "Generation is becoming real-time. Verification has to become real-time too."
