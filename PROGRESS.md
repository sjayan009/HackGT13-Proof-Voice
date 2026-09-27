# PROGRESS.md — ProofVoice / HackGT 13

Living log. Every phase: status · measured metric · blocker · next step. Newest notes at the bottom of each section.

## ⚠️ Open decision for the human (score orientation)

The organizer scorer (`HackGTMinDCF.zip` → ASVspoof5 `evaluation-package`, `calculate_modules.compute_det_curve`)
treats **bona fide as the target class: a HIGHER `cm-score` = MORE BONA FIDE**. The HEARSAY instructions / our
docs say `cm-score` is a synthetic probability (1.0 = synthetic). These are opposite orientations.
If the organizers run their script unmodified on a p_synthetic file, our ranking would be inverted and minDCF would be
catastrophic. `ml/generate_hearsay_tsv.py` therefore writes the HEARSAY-convention file (1.0 = synthetic) as the
primary output **and** a sibling `*_bonafide_high.tsv` (= 1 − p). **Ask the organizers which one they score** (or
use the one-time review to confirm) before final submission. All internal minDCF numbers below are computed with the
official code on correctly oriented scores (`ml/evaluate_mindcf.py` converts p_synth → 1 − p_synth).

## Phase 0 — Organizer package audit — COMPLETE

Hardware: RTX 4060 Laptop 8 GB (WDDM; batches must stay < ~7 GB or they spill to shared memory and crawl),
22 logical CPUs, 31 GB RAM, 480 GB free disk. Python 3.11.9 venv at `.venv` (torch 2.6.0+cu124). Node present.
Docker CLI present. **No system ffmpeg** → `imageio-ffmpeg` binary used locally; Docker image installs ffmpeg.

| Item | Location | Verified |
|---|---|---|
| DiffSSD | `data/hearsay/train/DiffSSD/generated_speech/` (already extracted) | 70,000 spoof clips, 10 generators, 0 unreadable |
| LJRealResampled | `data/hearsay/train/ljreal/` | **242** bona fide WAVs, 16 kHz, 0.47 h |
| Held-out test | `data/hearsay/test/HackGTHearsayTesting.zip` (untouched) → extracted copy `data/hearsay/test/extracted/HackGTHearsayTesting/` | 1,671 WAV, all 16 kHz mono PCM16, 3.02–13.58 s, median 3.41 s |
| Template | `data/hearsay/template/HearsayScoreKey4TeamX.tsv` (never written) | 1,671 rows, `filename\tcm-score`, all 0.5 |
| Scorer | `data/hearsay/scoring/HackGTMinDCF/…/evaluation-package` (+ verbatim copy `ml/official_scoring/`) | runs; reproduces shipped `track1_result.txt` |
| Also shipped | ASVspoof5 Baseline-AASIST **with pretrained `best.pth`**, Baseline-RawNet2 code (no weights) | |

DiffSSD generators (sr / format / speakers / clips): diffgan_tts, grad_tts, pro_diff, wavegrad2 (22.05 kHz PCM, **LJ voice**, 5k each);
elevenlabs (44.1 kHz **MP3**), playht (24 kHz **MP3**), xtts_v2 (24 kHz), unit_speech (22.05 kHz), your_tts (16 kHz),
openvoicev2 (22.05 kHz, 5 accents × 5k) — these six clone **10 LibriSpeech speaker IDs** (speaker_100, _1487, _2061, …).
Full per-generator stats: `outputs/dataset_audit.json`.

## Phase 1 — Official minDCF — COMPLETE

Organizer modification of ASVspoof5 Track 1: `Pspoof 0.05→0.5`, `Cfa 10→4`, `Cmiss 1`. Normalised:
**minDCF = min_t [ FRR_bonafide(t) + 4 · FAR_spoof(t) ]** (range 0 … 1; 1.0 = no better than rejecting everything).
Note: in the official code the 4× cost falls on *accepting a spoof as bona fide*; the dataset prior (~70 % real) does not
enter minDCF (rates only).

- `ml/evaluate_mindcf.py` — imports organizer `calculate_modules` verbatim; CLI mirrors `evaluation.py --m t1`,
  plus `--orientation synthetic_high`.
- `ml/tests/test_mindcf_parity.py` — 5 tests pass: shipped fixture regression (0.857142857…), subprocess parity vs
  organizer `evaluation.py`, fast-path equality, orientation guard, cost-model guard.

## Phase 2 — Dataset audit + frozen manifests — COMPLETE

**Blocker found → mitigated:** only 242 bona fide clips (one speaker) vs 70k spoofs, and 60 % of spoof generators clone
*LibriSpeech* voices. A detector trained only on LJ-real would learn "not Linda Johnson ⇒ synthetic". Added public
**LibriSpeech dev-clean + test-clean** (CC BY 4.0, 5,323 utterances, 80 speakers, native 16 kHz) as extra bona fide.
Deliberately **not** added: the rest of LJSpeech (held-out real clips may be drawn from it → contamination risk).

`ml/make_splits.py` (seed 1337) → `outputs/manifests/{train,val}.csv`, group-disjoint:
LJ by chapter, LibriSpeech by speaker, single-speaker spoofs by sentence id, multi-speaker spoofs by cloned speaker
(val speakers 2061, 6167). Spoofs capped 2,500 train / 600 val per generator.
Train 29,444 (4,444 bona fide) · Val 7,121 (1,121 bona fide: 1,077 Libri + **44 LJ**) — LJ val is small; LJ-specific
minDCF is reported separately and should be read with that caveat.

Audio cache: `ml/audio_cache.py` decodes everything once to 16 kHz int16 with **one** resampler (soxr_hq) for all classes.
Validation uses seeded **test-like crops (3.0–4.6 s)** matching the held-out duration regime.

## Phase 3 — Baseline detector — IN PROGRESS

| Model | Val pooled minDCF | AUC | EER | notes |
|---|---|---|---|---|
| AASIST (organizer ASVspoof5 ckpt), zero-shot | **1.000** | 0.638 | 44.0 % | fails on DiffSSD; ok only on diffgan/your_tts/openvoicev2 |

| Hand-crafted forensic features (38, LR, train→val) | 0.506 | 0.900 | 17.6 % | spectral alone 0.570; splice/compression ≈ chance for detection |
| XLS-R-300M (first 12 layers) fine-tuned, step 500 | **0.314** | 0.976 | 9.5 % | LJ-voice subset 0.12; cloned-Libri subset 0.34 |

Engineering blockers hit & fixed: full 24-layer XLS-R + AdamW exceeds 8 GB (WDDM spills to shared memory → 10× slower)
→ truncated to first 12 transformer layers; cuDNN conv workspace failures at eval → eval without cuDNN, per-process
memory cap 90 %; frozen-embedding probe (`ml/extract_ssl.py`/`probe.py`) OOM'd and was deprioritised in favour of
fine-tuning (scripts kept; not part of results).

Error analysis at step 500: hardest spoofs are **elevenlabs (0.47) and playht (0.44)** clones of the two *held-out*
speakers → part of the model's power is speaker identity (only 8 cloned voices in training). **LJ bona fide scores
higher than Libri bona fide** (p95 0.089 vs 0.005) → if held-out reals are LJ, false alarms come from there.
Next: `--lj-share` bona fide sampling (v2), unseen-generator run, AASIST fine-tune (family 2).

**Phase 3 exit gate MET** (v1): val minDCF 0.1534 via serving path, TSV generated + validated (1,671 rows).

| Run | change | best val minDCF | EER | AUC | notes |
|---|---|---|---|---|---|
| xlsr12_v1 | baseline recipe | 0.1534 (step 2500) | 4.35 % | 0.9932 | LJ bona fide p99 0.0135 |
| xlsr12_v2_lj30 | 30 % of bona fide draws from LJ | **0.1311** (step 1500) | 3.02 % | 0.9961 | LJ p99 0.0015; better unit_speech/xtts/your_tts; elevenlabs still 0.34 |
| v1+v2 logit average | ensemble | 0.1347 | | | not better than v2 → not used |

Checkpoint-to-checkpoint noise within a run is ~±0.03 minDCF, so v1→v2 is suggestive, not conclusive.
**Selected: v2** (`models/selected`). Held-out sanity (no labels): v2 flags 45 % of held-out at its val threshold
(v1: 33 %; organizers state ~30 % synthetic) → the val-optimal *threshold* transfers imperfectly; ranking between v1
and v2 agrees (Spearman 0.94, top-500 overlap 466). minDCF is threshold-free, so the TSV is unaffected; the UI
status threshold may be aggressive on out-of-domain audio (documented limitation).

Integration found: full-length 9.7 s LJ bona fide clip scored 0.56 and 2 s windows up to 0.99 → out-of-regime
inputs. Timeline windows raised to 3 s (≥ held-out min); v3 trains on 2.5–6 s crops + 5,803 extra train-only
bona fide (LibriSpeech dev-other/test-other); `ml/aggregation.py` measures long-clip aggregation.

Phase 5 (forensic breadth) — branch standalone + fusion ablation in `outputs/results/fusion.json`:
weighted-logit fusion, weight chosen on 4 folds, evaluated on the 5th (mean per-fold minDCF):
primary 0.068 · +all branches 0.064 · +spectral 0.065 · +prosody 0.066 · +quality 0.066 · +splice 0.070 ·
+compression 0.077. Gains ≤ 0.004 with unstable weights → **score path = primary only**; branches = evidence/routing.
(An earlier OOF-refit protocol was discarded: its per-fold recalibration alone cost 0.15 minDCF.)

Phase 10 Grok red team: verified in the browser end-to-end — live Grok Voice → same detector → 83 % synthetic,
likely_synthetic, analysis confidence 0.97, source honestly labelled "Grok Voice (realtime)".
The 10 cloned speakers are all LibriSpeech **train-clean-360** (23 GB); deliberately not downloaded (size, and
targeted overlap with possible held-out reals).

## Subagent workstreams (Sonnet) — delivered

- F frontend `web/` (Next.js 14; `npm run build` passes; no fake numbers; eval drawer shows "not computed" on 404).
- D forensic branches `api/app/forensics/` (metadata, spectral, prosody, compression, splice, quality; 56 tests pass).
- G xAI `api/app/services/grok.py` — Grok Voice realtime (`wss://api.x.ai/v1/realtime`, model grok-voice-latest)
  **verified live**; xAI TTS fallback verified; cached fixture `api/app/fixtures/grok_fixture.wav`; 5 real Grok samples
  in `data/grok_samples/` for OOD testing; 8 tests pass; key never written anywhere.

## Phase 4 — model race — COMPLETE (within time budget)

| Run | change vs previous | best val minDCF | EER | AUC |
|---|---|---|---|---|
| AASIST organizer ckpt, zero-shot | — | 1.000 | 44.0 % | 0.638 |
| hand-crafted branches (LR) | — | 0.506 | 17.6 % | 0.900 |
| xlsr12_v1 | fine-tuned XLS-R (12 L) | 0.153 | 4.4 % | 0.993 |
| xlsr12_v2_lj30 | + 30 % LJ bona fide draws | 0.131 | 3.0 % | 0.996 |
| **xlsr12_v3 (SELECTED)** | + train-only LibriSpeech dev/test-other bona fide, 2.5–6 s crops | **0.066** | **1.5 %** | **0.999** |

v3 beats v2 at every checkpoint (0.084/0.112/0.098/0.066/0.076/0.076), well beyond the ±0.03 checkpoint noise.
Per-generator (v3): 0.000 for 7 generators; unit_speech 0.007, playht 0.062, **elevenlabs 0.207** (weakest; val
elevenlabs clips are clones of speakers never seen in training). Calibration ECE 0.015.
Held-out sanity (no labels): v3 flags 30.8 % of held-out clips at its validation threshold ≈ organizers' ~30 % prior.
Not done (time): AASIST fine-tune (`ml/train_aasist.py` ready), unseen-generator retrain (`ml/unseen.py` ready).

## Phase 5/6 — forensic breadth + orchestration — COMPLETE
Fusion re-run against v3: primary 0.0332 per-fold; no branch improves it (chosen weights mostly 0) →
score = primary only; spectral/prosody/compression/splice/quality/metadata are routed evidence (branch_log explains).

## Phase 7 — robustness — COMPLETE (`outputs/results/robustness.json`, n=1400)
clean 0.054 · AAC-48k 0.073 · Opus-24k 0.096 · MP3-64k 0.123 · clipping 0.127 · MP3→Opus→MP3 0.133 ·
μ-law 8 kHz 0.133 (+ noise/gain rows in the file). MP3 mainly raises bona fide false alarms (FRR 2.6 %→14.7 % at the
clean threshold): two MP3-sourced spoof generators taught "MP3 artefacts ⇒ synthetic". Next fix: MP3/Opus codec
augmentation on bona fide during training.

## Phase 10 — Grok red team — COMPLETE
Live Grok Voice → same detector verified in browser (83 %, likely_synthetic, v1). OOD with v3: **6/6 real Grok Voice
clips flagged** (p 0.91–1.00) — a generator never seen in training (`outputs/results/grok_ood.json`).
Latency: 23 ms per 4 s clip on RTX 4060 fp16, 345 ms on CPU (`latency.json`).

## Phase 13 — Docker — image builds; offline TSV run verified (see HANDOFF for result)

## Phase 8 — temporal aggregation — COMPLETE
Full-length val subsample (n=2000): mean of 4 s window logits 0.016 vs full clip 0.021 vs median 0.023 vs max 0.026
→ clips > 4 s now scored as the mean of 4 s windows (`PrimaryDetector.clip_logits`, used by API and TSV).
Final TSV regenerated with this rule (31.4 % of held-out above threshold).
