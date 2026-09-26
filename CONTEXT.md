# CONTEXT.md — ProofVoice / HackGT 13

## Mission

Build **ProofVoice**, a real-time evidence layer for voice authenticity, as one coherent submission targeting:

1. **Oracle of the Deep** — selected HackGT track
2. **NSA HEARSAY: The Audio Authentication Challenge** — primary scientific target
3. **Meta: Bringing People Closer Together with AI**
4. **SpaceXAI: Make it Legendary with SpaceXAI**

This is a solo hackathon build. Optimize for:
- measurable detection performance,
- scientific credibility,
- forensic breadth,
- live-demo impact,
- sponsor fit,
- reliability.

Do not optimize for raw feature count.

---

## Official HEARSAY facts

The official challenge requires a system that:

- accepts common audio formats,
- autonomously analyzes the audio,
- combines multiple forensic techniques,
- outputs a synthetic probability from `0.0` to `1.0`,
- provides explainable results,
- processes a held-out test set,
- submits a tab-delimited prediction file,
- includes source code,
- includes a Docker container reproducible by judges.

The official scoring framework emphasizes:

- **60% detection performance**
- **20% forensic diversity / innovation / technical depth**
- **20% documentation / presentation**

The organizer instructions further state:

- approximately **70% of held-out files are real / bona fide**
- the scoring scenario is analyst-facing
- a false alarm is described as costing **4× more** than a missed detection
- detection performance uses **ASVspoof 5 Track 1 minDCF**
- lower minDCF is better
- NSA offers an optional **one-time review** of a draft prediction TSV before final submission

Therefore:

> Do not optimize ordinary accuracy or balanced accuracy. Optimize the official challenge metric.

---

## Audited official held-out test set

The official testing package contains:

- **1,671 WAV files**
- all **16,000 Hz**
- all **mono**
- all **16-bit PCM**
- clips approximately **3.0–13.6 seconds**
- median duration approximately **3.4 seconds**

The official score template contains:

- `filename`
- `cm-score`

with exactly 1,671 rows.

`cm-score` semantics:

- `0.0` = confidently real
- `1.0` = confidently synthetic

Never remove rows. Never invent filenames. Never use the held-out set for training or pseudo-labeling.

---

## Official training sources

The organizers provide:

### Bona fide
- `LJRealResampled.zip`
- derived from LJSpeech / Linda Johnson speech

### Synthetic
- `DiffSSD.zip`
- large Purdue synthetic-speech corpus
- includes multiple TTS families including ElevenLabs and diffusion-based synthesis

The official Google Drive folder also contains:

- `HackGTHearsayTesting.zip`
- `HearsayScoreKey4TeamX.tsv`
- `HackGTMinDCF.zip`
- official HEARSAY instructions

Use the organizer-provided `HackGTMinDCF.zip` implementation as the authoritative local scoring reference. Validate any reimplementation against it.

---

## Core research conclusion

The pre-hackathon research converged on one central finding:

> **Real-time synthesis and conversational speech are advancing faster than open-world audio authenticity verification.**

Modern speech systems already support:
- native speech-to-speech,
- streaming STT/TTS,
- interruptions / barge-in,
- expressive prosody,
- voice replication,
- full-duplex interaction,
- tool use during live voice sessions.

The less-solved problems are:
- unseen generators,
- neural-codec shifts,
- transcoding / laundering,
- replay,
- noise,
- channel changes,
- partial manipulation,
- calibration under distribution shift,
- real-time inference from incomplete evidence.

ProofVoice attacks that gap.

---

## Product thesis

### One sentence

**ProofVoice detects, localizes, and explains evidence of synthetic speech in files and live audio while explicitly representing uncertainty.**

### Human impact

AI voice cloning damages one of the oldest trust signals in communication: recognizing someone by their voice.

ProofVoice is not trying to replace communication. It exists to help people reason about whether the voice they hear may be synthetic.

---

## Two product modes

### 1. Forensic Lab

For NSA and Oracle.

Input:
- WAV
- MP3
- M4A
- OGG
- MP4 audio
- other ffmpeg-decodable formats

Output:
- synthetic probability
- analysis confidence / evidence sufficiency
- detector-level evidence
- suspicious time segments
- file / codec / metadata analysis
- optional manipulation indicators
- exact NSA-compatible TSV generation

### 2. Live Trust

For Meta, Oracle, and SpaceXAI.

Input:
- microphone stream
- Grok Voice stream

Output:
- rolling synthetic probability
- time-to-confidence
- evidence confidence
- suspicious timeline
- detector disagreement
- channel-quality indicators

---

## Sponsor fit

### NSA HEARSAY

This is the scientific core.

The system must satisfy the official challenge:
- multiple forensic techniques,
- real/synthetic probability,
- explainability,
- batch inference,
- official TSV,
- Docker reproducibility.

### Oracle of the Deep

Oracle is the visualization / ML depth surface.

Show:
- temporal probability evolution,
- minDCF,
- detector ablations,
- open-generator validation when possible,
- robustness under re-encoding/noise,
- time-to-confidence,
- score distributions.

### Meta

The Meta story is human-to-human trust:

> Synthetic speech can make people doubt real voices. ProofVoice gives people evidence and uncertainty so trusted communication remains possible.

Do not turn the product into an AI companion.

### SpaceXAI

Grok must be structurally meaningful.

Primary integration:

```text
Grok Voice
   ↓
live synthetic audio
   ↓
ProofVoice forensic pipeline
   ↓
score timeline + time-to-confidence
```

Secondary optional integration:
- grounded explanation of structured forensic evidence

Grok must not set the classifier score.

---

## Forensic breadth strategy

NSA explicitly values distinct analysis families.

Target branches:

1. **Deep-learning anti-spoof detector**
2. **Spectral / frequency-domain analysis**
3. **Prosody / phonetics**
4. **Compression / transcoding / codec analysis**
5. **Container / metadata analysis**
6. **Speaker-embedding consistency**
7. **Splice / discontinuity analysis**
8. **Acoustic-environment consistency**, only if practical

Do not force every branch into the final probability.

A branch can still earn value as:
- supporting evidence,
- routing information,
- explanation,
- quality control.

---

## Agentic orchestration

NSA explicitly rewards choosing analyses based on context rather than brute-forcing everything.

ProofVoice should implement deterministic evidence-based orchestration first.

Example:

```text
inspect file
   ↓
codec / metadata / duration / channel quality
   ↓
run primary anti-spoof detector
   ↓
if confidence is weak or codec is suspicious:
    run deeper spectral / compression / splice analyses
   ↓
if local anomaly appears:
    analyze that region more deeply
```

An LLM may summarize evidence, but should not hallucinate forensic findings.

---

## Metric strategy

Primary:
- official ASVspoof 5 Track 1 minDCF

Also track:
- ROC-AUC
- EER
- F1
- false-positive rate
- false-negative rate
- log loss
- Brier score / ECE if practical
- inference latency
- time-to-confidence

Because false alarms are costly:
- inspect bona fide score distribution carefully
- do not tune only for synthetic recall

---

## Validation strategy

If DiffSSD provides generator labels:

- grouped train/validation splits
- leave-one-generator-out or held-out-generator validation
- separate random stratified split for iteration

Never claim open-world robustness from a random split alone.

---

## Robustness tests

Minimum:
- MP3 re-encoding
- Opus re-encoding
- additive noise
- gain changes
- repeated encode/decode

If time permits:
- telephony band-limiting
- replay approximation
- reverberation
- clipping
- splice edits

Measure impact on minDCF.

---

## Important architecture consequence

The official held-out test set is entirely 16 kHz.

For NSA scoring:
- optimize around 16 kHz
- do not spend competition time building high-frequency forensic branches that cannot be present in the held-out audio

For the broader public/live product:
- preserve higher-rate input if easy
- resample to the selected forensic model rate

---

## Scientific hypothesis

> **A calibrated combination of complementary forensic evidence, evaluated using the official minDCF metric and realistic laundering transforms, can outperform a naive single-model detector while remaining interpretable enough for real-time trust decisions.**

This is a hypothesis, not a guaranteed result.

If fusion does not improve validation minDCF, use the best single detector and keep other forensic branches as explainable evidence.

---

## Truthfulness rules

Never claim:
- universal deepfake detection
- 100% accuracy
- no watermark = human
- speaker similarity = liveness
- metadata = proof
- one generator fingerprint generalizes to every synthesis method

Preferred language:
- synthetic probability
- evidence suggests
- inconclusive
- suspicious segment
- detector disagreement
- analysis confidence
- provenance evidence

---

## Must ship

- official minDCF evaluator or validated wrapper around organizer code
- strongest anti-spoof detector found within time
- calibrated / well-scaled `cm-score`
- NSA TSV generator
- TSV validator
- Docker container
- file analyzer
- Trust Timeline
- live mode
- Grok red-team mode
- reproducible benchmark results
- README
- demo video
- public repository

---

## Optional only after core is green

- second/third detector fusion
- manipulation subtype classifier
- speaker consistency
- AudioSeal provenance
- replay-specific classifier
- Grok explanation layer
- sophisticated acoustic-scene analysis

---

## Winning definition

The finished project should have:

- one strong measurable metric story,
- one robust inference pipeline,
- one memorable live demo,
- one clear human-impact story,
- one honest limitations section,
- one reproducible Docker path,
- one submission TSV validated against all 1,671 filenames.

Priority order:

1. NSA detection/minDCF
2. NSA forensic breadth
3. Oracle ML + visualization
4. Meta human-trust framing
5. SpaceXAI Grok integration
6. optional polish
