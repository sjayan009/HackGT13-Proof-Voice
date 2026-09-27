# NSA HEARSAY — ProofVoice final submission

**Everything a HEARSAY judge needs is in this folder or linked from it.**

| File | What it is |
|---|---|
| **[`ProofVoice_final.tsv`](ProofVoice_final.tsv)** | **Final predictions for all 1,671 held-out test files.** `cm-score` = probability the clip is **synthetic** (1.0 = synthetic, 0.0 = real), as the challenge instructions specify. |
| [`ProofVoice_final_bonafide_high.tsv`](ProofVoice_final_bonafide_high.tsv) | The same scores flipped (`1 − p`), for a scorer that expects **higher = bona fide** (the ASVspoof5 `calculate_metrics.py` convention). Use one or the other, not both. |
| **[▶ Demo video on YouTube](https://youtu.be/hpHuJdQuLgU)** · [`ProofVoice_demo_video.mp4`](ProofVoice_demo_video.mp4) | 1:40 demo of the full system (file analysis, live streaming, Grok Voice red team). |

Both TSVs have the organizer template's exact header (`filename`, `cm-score`) and filename set, and pass
[`ml/validate_hearsay_tsv.py`](../ml/validate_hearsay_tsv.py). 31.4% of test files score above the decision
threshold, consistent with the stated ~30% spoof share.

## Validation performance (lower minDCF is better)

Computed with the organizer's own ASVspoof5 functions on a group-disjoint validation split (1,121 bona fide / 6,000
spoof). Reproduce with `python ml/rescore_nsa_settings.py` → [`outputs/results/mindcf_nsa_settings.json`](../outputs/results/mindcf_nsa_settings.json).

| Model | **minDCF, NSA final settings** (Pspoof 0.3, Cmiss 1, Cfa 4) | minDCF, HackGTMinDCF.zip (Pspoof 0.5) | EER |
|---|---|---|---|
| **XLS-R v3 (submitted)** | **0.0393** | 0.0656 | **1.53%** |
| XLS-R v4 (codec + hard-generator augmentation) | 0.0454 | 0.0799 | 1.77% |
| XLS-R v2 | 0.0780 | 0.1311 | 3.02% |
| XLS-R v1 | 0.1080 | 0.1534 | 4.36% |
| AASIST, fine-tuned | 0.3124 | 0.4450 | 11.95% |
| AASIST organizer checkpoint, zero-shot | — | 1.000 | 43.98% |

minDCF picks its own threshold, so changing Pspoof changes the reported number but not the submitted scores, and
v3 is the best model under both settings.

## How the TSV was produced (reproducible)

```bash
# Docker, fully offline (no internet, no xAI key)
docker build -t proofvoice .
docker run --rm -v "$PWD/data/hearsay:/data" -v "$PWD/outputs:/out" proofvoice

# or directly
python ml/generate_hearsay_tsv.py --input data/hearsay/test \
    --template data/hearsay/template/HearsayScoreKey4TeamX.tsv --output outputs/team_predictions.tsv
```

**Model weights** (`models/selected/`, 315 MB) exceed GitHub's file limit, so they are attached to the
[`hearsay-final` release](https://github.com/sjayan009/HackGT13-Proof-Voice/releases/tag/hearsay-final). Unzip
`proofvoice_model_selected_v3.zip` into the repo root (it contains `models/selected/`) before building Docker.
No test-set labels, pseudo-labels or test audio were used in training.

## Where to look next

| Rubric area | Start here |
|---|---|
| Detection performance (60%) | Table above · [RESULTS.md](../RESULTS.md) (auto-generated from result files) |
| Forensic diversity / depth (20%) | [README → Forensic techniques](../README.md#forensic-techniques-distinct-families) · code in [`api/app/forensics/`](../api/app/forensics/) and [`api/app/detectors/`](../api/app/detectors/) |
| Documentation (20%) | [README](../README.md) · [MODEL_STRATEGY.md](../MODEL_STRATEGY.md) (subgroup + domain-shift audit) · [HEARSAY_DATA_AUDIT.md](../HEARSAY_DATA_AUDIT.md) · [PROGRESS.md](../PROGRESS.md) |
| Training + evaluation code | [`ml/`](../ml/) (splits, training, calibration, minDCF parity test, robustness, TSV generation) |
