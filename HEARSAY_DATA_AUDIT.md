# HEARSAY_DATA_AUDIT.md

## Official Google Drive package

The organizer-provided folder contains:

- `DiffSSD.zip` — ~18.11 GB
- `LJRealResampled.zip` — ~46.25 MB
- `HackGTHearsayTesting.zip` — ~169.83 MB
- `Copy of HearsayScoreKey4TeamX.tsv`
- `HackGTMinDCF.zip` — ~6.97 MB
- `HEARSAY_HackGT2026_Instructions.pdf`

## Training

### Real / bona fide
`LJRealResampled.zip`

### Synthetic
`DiffSSD.zip`

Organizer instructions state DiffSSD includes multiple TTS outputs including ElevenLabs and diffusion-based systems.

## Held-out test

Verified from the provided archive:

- 1,671 files
- WAV
- 16 kHz
- mono
- 16-bit PCM
- shortest ~3.02 s
- median ~3.41 s
- mean ~3.69 s
- longest ~13.58 s

## Prediction template

Verified:

- 1,671 rows
- columns:
  - `filename`
  - `cm-score`
- current placeholder scores are uniform
- filenames correspond to the test files

## Scoring

Organizer instructions specify:

- ~70% of held-out files are real
- analyst scenario
- false alarms are weighted more heavily
- false positive cost described as 4× a miss
- ASVspoof 5 Track 1 minDCF
- 60% detection performance
- 20% creativity / quality / innovation / depth
- 20% documentation
- lower minDCF is better

The official package includes `HackGTMinDCF.zip`.

Use that package as the authority.

## One-time review

NSA offers one optional draft prediction-file review.

Do not spend it on the first baseline.

Use it after:
- model selection
- fusion decision
- exact TSV validation

## Important architecture consequence

The official held-out test set is entirely 16 kHz.

Prioritize 16 kHz anti-spoofing performance for HEARSAY.

Do not burn competition time on high-frequency features that cannot exist above the 8 kHz Nyquist limit in this test set.

## Recommended local layout

```text
data/hearsay/
├── train/
│   ├── diffssd/
│   └── ljreal/
├── test/
│   └── HackGTHearsayTesting.zip
├── scoring/
│   └── HackGTMinDCF/
└── template/
    └── HearsayScoreKey4TeamX.tsv
```
