# ARCHITECTURE.md — ProofVoice

## 1. Design goal

One forensic core powers:

- offline HEARSAY inference
- batch test-set scoring
- live microphone analysis
- Grok Voice red-team analysis
- Oracle visualizations
- Meta trust experience

There must not be separate fake "demo logic."

---

## 2. High-level architecture

```text
                     INPUTS
        ┌──────────────┼──────────────┐
        │              │              │
     File Upload   Live Mic      Grok Voice
        │              │              │
        └──────────────┴──────────────┘
                       │
                       ▼
              AUDIO INGESTION
              ffmpeg / torchaudio
                       │
                       ▼
             INITIAL FORENSIC SCAN
        codec / metadata / duration / quality
                       │
                       ▼
              PRIMARY ANTI-SPOOF MODEL
                       │
            ┌──────────┴──────────┐
            │                     │
      confident result       ambiguous / suspicious
                                  │
                                  ▼
                    DEEP FORENSIC BRANCHES
            ┌────────────┬────────────┬────────────┐
            │            │            │            │
        spectral      prosody     compression     splice
            │            │            │            │
            └────────────┴──────┬─────┴────────────┘
                                │
                                ▼
                       FUSION / CALIBRATION
                                │
                                ▼
                      TEMPORAL AGGREGATION
                                │
               ┌────────────────┼────────────────┐
               ▼                ▼                ▼
          HEARSAY TSV       Judge UI        Live Timeline
```

---

## 3. Repository layout

```text
/
├── CONTEXT.md
├── ARCHITECTURE.md
├── PHASES.md
├── CLAUDE_OPUS_5_5_PROMPT.md
├── HEARSAY_DATA_AUDIT.md
├── README.md
├── .gitignore
├── .env.example
│
├── web/
│   ├── app/
│   ├── components/
│   ├── lib/
│   ├── public/
│   ├── package.json
│   └── .env.local
│
├── api/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── schemas.py
│   │   ├── audio/
│   │   ├── detectors/
│   │   ├── forensics/
│   │   ├── fusion/
│   │   ├── orchestration/
│   │   ├── services/
│   │   └── tests/
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── .env
│
├── ml/
│   ├── inspect_data.py
│   ├── make_splits.py
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_mindcf.py
│   ├── calibrate.py
│   ├── benchmark.py
│   ├── ablate.py
│   ├── generate_hearsay_tsv.py
│   └── validate_hearsay_tsv.py
│
├── data/
│   └── hearsay/
│       ├── train/
│       │   ├── diffssd/
│       │   └── ljreal/
│       ├── test/
│       ├── scoring/
│       └── template/
│
├── models/
├── outputs/
├── tmp/
├── scripts/
└── Dockerfile
```

---

## 4. Official HEARSAY data contract

### Test set

- 1,671 WAV files
- 16 kHz
- mono
- PCM16
- ~3.0–13.6 s

### Submission format

Tab-separated:

```text
filename    cm-score
example.wav 0.8123
```

`cm-score`:
- 0 = real
- 1 = synthetic

Output validator must enforce:
- 1,671 rows
- exact filename set
- no duplicates
- no missing rows
- numeric score
- `[0,1]`
- tab-separated output
- header exactly as expected

Never overwrite the organizer template.

---

## 5. Training data

Organizer-provided:

- `LJRealResampled.zip`
- `DiffSSD.zip`

Expected layout after extraction:

```text
data/hearsay/train/
├── ljreal/
└── diffssd/
```

Before training:
- inventory files
- inspect labels
- inspect generator metadata
- inspect duration/sample-rate distributions
- inspect train-set leakage risks

Do not assume file structure before inspection.

---

## 6. Official scoring

Use organizer-provided:

```text
HackGTMinDCF.zip
```

Create a local wrapper but preserve the organizer logic.

Required tests:

1. known fixed score fixture
2. compare wrapper result against organizer script
3. regression test to prevent accidental metric drift

Primary model-selection metric:
- **minDCF**

Secondary:
- ROC-AUC
- EER
- FPR/FNR
- log loss
- latency

---

## 7. Audio ingestion

### File path

Use ffmpeg/ffprobe for robust decoding.

Extract:
- container
- codec
- duration
- sample rate
- channel count
- bit depth if known
- encoder tags
- timestamps where present
- unusual metadata

Normalize inference audio to:
- mono
- float32
- 16 kHz

Because the held-out test set is already 16 kHz, the main competition path is 16 kHz-native.

---

## 8. Primary anti-spoof model

Select empirically.

Candidate families:
- AASIST
- RawNet2 / RawNet3
- wav2vec2 / WavLM / HuBERT classifiers
- ASVspoof 5 baselines
- strong public deepfake-detection checkpoints

Required interface:

```python
class Detector:
    def score(self, audio: np.ndarray, sr: int) -> DetectorResult:
        ...
```

Return:
- raw score/logit
- synthetic probability if calibrated
- latency
- optional embedding
- diagnostics

---

## 9. Forensic branches

### A. Metadata / container

Use:
- ffprobe
- MediaInfo if installed
- ExifTool if installed

Potential evidence:
- encoder tag anomalies
- codec/container mismatch
- suspicious timestamps
- impossible or inconsistent metadata
- multiple encoding signatures when detectable

Metadata is evidence only, never proof.

### B. Spectral

Potential features:
- LFCC
- log-mel statistics
- CQT
- spectral flatness
- rolloff
- band-edge behavior
- harmonic structure

### C. Prosody / phonetics

Potential features:
- F0 contour
- pause timing
- voicing ratio
- energy variation
- pitch variance
- monotonicity
- breath/pause proxies

### D. Compression

Potential evidence:
- double-encoding artifacts
- transcoding fingerprints
- band-limiting
- spectral quantization

### E. Speaker consistency

Use ECAPA/x-vector style embeddings only if helpful.

Goal:
- detect within-clip drift

Do not call this liveness.

### F. Splice / discontinuity

Potential cues:
- abrupt background shifts
- DC offset changes
- phase discontinuities
- spectral seams
- local score jumps

---

## 10. Agentic orchestration

Implement deterministic routing.

Example:

```python
initial = inspect_file(file)

primary = primary_detector.score(audio)

if initial.codec_is_lossy:
    run_compression_branch()

if primary.confidence < threshold:
    run_spectral_branch()
    run_prosody_branch()

if temporal_anomaly_found:
    run_splice_branch(on_local_region)

if speaker_drift_possible:
    run_speaker_consistency()
```

Log every branch that ran and why.

This directly supports NSA's orchestration bonus.

---

## 11. Fusion

Start with primary detector only.

Then benchmark:
- primary
- primary + spectral
- primary + metadata
- primary + compression
- primary + speaker consistency
- combinations

Use simple fusion:
- logistic regression
- calibrated stacking
- weighted logit fusion

Optimize validation minDCF.

If fusion does not improve:
- do not force it.

---

## 12. Calibration

Possible:
- Platt scaling
- temperature scaling
- isotonic regression if validation data is large enough

Important:

If minDCF is computed by threshold sweeping, monotonic calibration may not improve minDCF.

Calibration still matters for:
- user-facing probabilities
- Meta trust UX
- interpretability

---

## 13. Temporal scoring

Even though HEARSAY ultimately needs one file score, the public product should localize suspicious regions.

Default starting point:

```text
window = 2.0 s
hop = 0.5 s
```

Benchmark alternatives.

Per-window schema:

```json
{
  "start_ms": 1000,
  "end_ms": 3000,
  "synthetic_probability": 0.82,
  "analysis_confidence": 0.73,
  "detector_scores": {},
  "evidence": []
}
```

Clip score aggregation candidates:
- mean
- median
- top-k mean
- percentile
- learned summary fusion

Choose using validation minDCF.

---

## 14. Live mode

Browser:
- Web Audio API
- AudioWorklet if stable
- binary PCM WebSocket

Server:
- rolling buffer
- resample to 16 kHz
- score overlapping windows
- emit JSON updates

Events:

```json
{
  "type": "analysis.window",
  "synthetic_probability": 0.84,
  "analysis_confidence": 0.76,
  "status": "likely_synthetic",
  "time_to_confidence_ms": 2400
}
```

Statuses:
- insufficient_evidence
- likely_human
- inconclusive
- likely_synthetic

---

## 15. Grok red-team

Preferred path:

```text
Grok Voice realtime
    ↓
capture streamed audio
    ↓
play to judge
    ↓
simultaneously feed ProofVoice
    ↓
score timeline
```

Use:
- `grok-voice-latest`

Keep `XAI_API_KEY` server-side.

Fallback:
- official xAI streaming TTS
- cached Grok-generated fixture

The core HEARSAY path must not depend on xAI or internet access.

---

## 16. API endpoints

### `GET /health`

Return:
- loaded detector
- device
- app version

### `POST /analyze/file`

Returns:
- clip score
- confidence
- timeline
- evidence
- branch execution log

### `WS /analyze/stream`

Input:
- PCM chunks

Output:
- rolling analysis events

### `POST /redteam/grok`

Starts a Grok synthetic test.

### `POST /explain`

Optional grounded explanation based only on structured evidence.

---

## 17. Frontend

Main screens:

### Forensic Lab
- file upload
- probability
- timeline
- suspicious regions
- evidence cards
- branch execution log

### Live Trust
- live probability
- time-to-confidence
- waveform
- timeline

### Red Team
- prompt
- Grok Voice generation
- live detector response
- compare human vs synthetic

### Evaluation drawer
- minDCF
- AUC
- EER
- robustness transforms
- ablation results

---

## 18. Docker

Docker is mandatory.

Container must support:

```bash
docker build -t proofvoice .
docker run ...
```

Required reproducible command:

```bash
python ml/generate_hearsay_tsv.py \
  --input data/hearsay/test \
  --template data/hearsay/template/HearsayScoreKey4TeamX.tsv \
  --output outputs/team_predictions.tsv
```

Do not require xAI for HEARSAY inference.

---

## 19. Tests

Must include:
- file decode
- 16 kHz normalization
- detector score bounds
- minDCF wrapper parity
- fusion serialization
- temporal aggregation
- TSV validator
- 1,671-row test
- WebSocket integration
- file-upload E2E
- Grok fallback path

---

## 20. Reliability priorities

If time is short:

Keep:
1. exact scoring
2. best detector
3. valid TSV
4. Docker
5. forensic timeline
6. live mode
7. Grok red team
8. README

Cut:
1. speaker verification
2. watermark integration
3. subtype classifier
4. weak fusion branches
5. Grok explanation
6. decorative UI
