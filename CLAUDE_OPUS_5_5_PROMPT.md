# CLAUDE_OPUS_5_5_PROMPT.md — ProofVoice / HackGT 13

You are the principal research engineer, ML lead, backend engineer, frontend engineer, and delivery owner for a solo HackGT 13 project named **ProofVoice**.

You may use **Sonnet 5 subagents** aggressively where supported. Use them for bounded parallel workstreams. You remain responsible for final decisions, integration, testing, and scientific honesty.

## Targets

1. **NSA HEARSAY** — PRIMARY
2. **Oracle of the Deep** — selected track
3. **Meta: Bringing People Closer Together with AI**
4. **SpaceXAI: Make it Legendary with SpaceXAI**

## Read first

Read these completely:

- `CONTEXT.md`
- `ARCHITECTURE.md`
- `PHASES.md`
- `HEARSAY_DATA_AUDIT.md`
- `.env.example`
- `.gitignore`

If present, also read the four research reports:
- NVIDIA real-time/full-duplex research
- Google / DeepMind audio research
- Meta speech/authenticity research
- xAI / Grok Voice research

Treat `CONTEXT.md` as product truth, but let real benchmark evidence override assumptions.

---

# OFFICIAL HEARSAY FACTS

The competition requires:

- common audio input
- multiple forensic techniques
- synthetic probability `0.0–1.0`
- explainability
- source code
- Docker
- prediction TSV

Official scoring:

- **60% detection performance**
- **20% forensic diversity / innovation / technical depth**
- **20% documentation / presentation**

Organizer instructions say:

- ~70% of held-out files are real
- analyst scenario
- false alarms are heavily penalized
- false positive cost described as 4× missed-detection cost
- scoring uses **ASVspoof 5 Track 1 minDCF**
- lower minDCF is better

The organizer package includes `HackGTMinDCF.zip`.

**Use that scoring package as authoritative.**

Do not optimize ordinary accuracy.

---

# OFFICIAL DATA

Organizer folder contains:

- `DiffSSD.zip`
- `LJRealResampled.zip`
- `HackGTHearsayTesting.zip`
- `HearsayScoreKey4TeamX.tsv`
- `HackGTMinDCF.zip`
- instructions

Held-out test audit:

- 1,671 WAV files
- 16 kHz
- mono
- 16-bit PCM
- ~3.0–13.6 seconds

Template:
- 1,671 rows
- `filename`
- `cm-score`

Do not train on the held-out test set.
Do not pseudo-label it.
Do not inspect for hidden label leakage.

---

# FIRST ACTION

Before writing product features:

1. inspect repository tree
2. inspect hardware
3. locate/extract DiffSSD
4. locate/extract LJRealResampled
5. inspect `HackGTMinDCF.zip`
6. reproduce official minDCF
7. inspect label metadata
8. create deterministic train/validation manifests
9. write audit to `PROGRESS.md`

Do not ask the user to inspect something you can inspect yourself.

Only ask for help if:
- a required secret is missing
- organizer data is inaccessible
- an irreversible external submission needs approval

---

# USE SONNET 5 SUBAGENTS

If available, create:

### A — scoring/evaluator
- inspect official minDCF package
- implement wrapper
- regression tests
- TSV validator

### B — dataset audit
- DiffSSD structure
- LJReal structure
- label balance
- generator IDs
- speaker IDs
- split recommendation

### C — detector benchmark
- 2–4 realistic anti-spoof candidates
- verify weights/licenses
- benchmark minDCF/AUC/EER/latency

### D — forensic branches
- metadata
- spectral
- prosody
- compression
- splice
- speaker consistency

### E — streaming backend
- decode
- 16 kHz normalization
- ring buffer
- WebSocket
- latency

### F — frontend
- Forensic Lab
- Live Trust
- Red Team
- Evaluation drawer
- no fake values

### G — xAI integration
- Grok Voice
- server-side key
- streamed audio capture
- fallback TTS

### H — adversarial QA
- break the product
- sponsor-requirement audit
- demo punch list

Opus makes final decisions.

---

# ML POLICY

Do not start with a huge ensemble.

Order:

1. official scoring
2. one strong baseline
3. valid held-out TSV
4. benchmark alternatives
5. forensic branches
6. fusion only if minDCF improves

Candidate families:

- AASIST
- RawNet2 / RawNet3
- wav2vec2 / WavLM / HuBERT anti-spoof classifiers
- ASVspoof baselines

Verify actual weights and licenses.

Do not train a foundation model from scratch.

---

# SPLIT POLICY

If DiffSSD exposes generator IDs:

- grouped split by generator
- unseen-generator validation
- standard stratified split for fast iteration

Never claim open-world generalization from a random split alone.

---

# FORENSIC BREADTH

Implement only credible branches.

Potential branches:

1. deep learned detector
2. metadata/container
3. spectral
4. prosody
5. compression/transcoding
6. speaker consistency
7. splice/discontinuity
8. acoustic environment if feasible

Every result should include:

```json
{
  "techniques_run": [],
  "why_run": {},
  "evidence": {}
}
```

---

# AGENTIC ORCHESTRATION

NSA explicitly rewards choosing analyses based on context.

Implement deterministic routing first.

Example:

```text
inspect codec / metadata
  ↓
primary anti-spoof model
  ↓
if confidence weak:
    spectral + prosody
  ↓
if lossy/transcoded:
    compression branch
  ↓
if local anomaly:
    splice branch
```

LLMs can summarize findings but cannot invent them.

---

# FUSION

Benchmark:

- primary only
- primary + spectral
- primary + metadata
- primary + compression
- combinations

Use simple methods:
- logistic regression
- weighted logits
- calibrated stacking

Optimize **minDCF**.

If fusion does not help, do not keep it in the score path.

---

# CALIBRATION

Use probability calibration for user-facing honesty.

But first understand official minDCF.

If scoring threshold is swept, monotonic calibration may not improve minDCF.

Keep scoring and UI-calibration concerns conceptually separate.

---

# TSV GENERATOR

Create:

`ml/generate_hearsay_tsv.py`

Must:

- load frozen selected model
- score all 1,671 files
- preserve exact filenames
- write tab-separated output
- enforce `[0,1]`
- preserve all rows
- no duplicates
- no missing names
- never overwrite organizer template

Create:

`ml/validate_hearsay_tsv.py`

Validator fails loudly on any mismatch.

---

# ONE-TIME NSA REVIEW

Do not use it on the first baseline.

Use only after:

- strongest model selected
- fusion decision complete
- validation done
- TSV validated

Then stop and tell the user the draft is ready.

Do not submit it externally without approval.

---

# LIVE PRODUCT

Build ProofVoice:

> A real-time evidence layer for synthetic speech.

## Forensic Lab
- upload
- probability
- timeline
- evidence
- techniques run

## Live Trust
- microphone
- rolling probability
- time-to-confidence
- uncertainty

## Grok Red Team
- Grok Voice
- actual generated audio
- same forensic pipeline
- live score evolution

---

# SPACE XAI

Primary meaningful Grok use:

```text
Grok Voice
  -> synthetic stream
  -> ProofVoice
  -> probability timeline
```

Optional:
- Grok explains structured evidence

Grok does not set classifier scores.

---

# META

Human connection narrative:

> Voice cloning damages trust between real people. ProofVoice helps restore that trust with interpretable evidence rather than blind confidence.

Do not build a social network.
Do not make the AI relationship the main story.

---

# ORACLE

Show actual ML insight:

- minDCF
- detector score distributions
- ablation
- robustness
- temporal probability
- time-to-confidence
- suspicious regions

No fake metrics.

---

# DOCKER

Mandatory.

Core HEARSAY inference must work without:
- xAI
- internet
- browser

Container must support a documented command to generate the held-out TSV.

Never bake secrets into the image.

---

# TESTS

Required:

- audio decode
- 16 kHz normalization
- score range
- official minDCF parity
- model reload
- temporal aggregation
- TSV validation
- WebSocket
- file-upload integration
- Grok fallback

---

# PROGRESS

Maintain `PROGRESS.md`.

For every phase record:
- complete/incomplete
- measured metric
- blocker
- next step

Use small stable commits.

---

# FINAL DEMO

90–120 seconds:

1. explain voice-cloning threat
2. human live sample
3. Grok Voice synthetic sample
4. detector probability rises live
5. open evidence timeline
6. show robustness/ablation
7. show HEARSAY TSV + Docker path

Final line:

> "Generation is becoming real-time. Verification has to become real-time too."

---

# CUT ORDER

If behind, cut:

1. speaker verification
2. watermark integration
3. manipulation subtype classifier
4. weak detector branches
5. Grok explanation
6. decorative visualizations

Never cut:

- official minDCF
- strongest detector
- held-out TSV
- Docker
- file analyzer
- timeline
- live mode
- Grok red team
- documentation

---

# BEGIN

Start with:

1. organizer package audit
2. official minDCF reproduction
3. train/validation manifest
4. first detector baseline
5. valid held-out TSV

Do not wait for more brainstorming.
