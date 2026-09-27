# Model strategy — evidence before promotion

## What the current headline misses

The selected XLS-R v3 reaches 0.0656 minDCF on the frozen validation split, but that split contains only **two held-out cloned speakers**. On ElevenLabs, speaker 2061 has 0.2415 subgroup minDCF and 62/296 misses at the global threshold; speaker 6167 has 0.0009 and no misses. PlayHT shows the same directional speaker gap. The two speakers have similar duration distributions, so clip length alone does not explain it. These numbers come from `ml/audit_generalization.py` and `outputs/results/generalization_audit.json`.

The bona fide errors are also uneven: 5/44 LJ clips trigger at the selected threshold versus 20/1,077 LibriSpeech clips. With only 44 LJ validation clips, the LJ rate is imprecise. A microphone or telephone recording is a further domain shift; the clean validation number does not certify it.

The held-out ElevenLabs voices differ acoustically as well as in score: cached forensic features show a mean spectral centroid of about 976 Hz for speaker 2061 versus 1,461 Hz for 6167, and estimated SNR of 16.5 versus 34.8 dB. Within speaker 2061, the missed clips are darker still (861 versus 1,007 Hz centroid for detected clips). These are observational differences, not a causal diagnosis; they make spectral/channel shortcuts a concrete hypothesis to test.

Channel stress is a separate failure. On the fixed 1,400-clip validation subset, MP3 64 kbps raises bona fide false alarms at the clean threshold from 2.6% to 14.7%, 20 dB noise raises them to 39.9%, and 10 dB noise to 90.4% (`outputs/results/robustness.json`). The existing waveform augmentation includes filtering and random noise, but did not include a real codec round trip. A paper on ASVspoof 5 augmentation reports that codec and compression conditions remain challenging even after laundering augmentation ([Ali et al.](https://arxiv.org/abs/2410.01108)); a cross-domain study also finds large effects from neural codecs ([Li et al.](https://aclanthology.org/2024.emnlp-main.286/)). Those studies support testing real channel transforms here; they do not establish that any candidate in this repo will improve.

## Training and evaluation decisions

1. Keep `models/selected` fixed until a candidate beats v3 on clean pooled minDCF **and** is checked on ElevenLabs/2061, LJ false alarms, MP3, and noise. A lower pooled score alone can hide a worse hard subgroup.
2. Fine-tune the organizer AASIST checkpoint with the same LJ share and extra train-only bona fide data as v3. This was run to step 600: minDCF improved from 0.487 at step 300 to 0.445, but remained far behind v3 (0.066), including ElevenLabs minDCF 0.620. The run was stopped after the second full validation pass and its checkpoint/results were retained in `models/aasist_v1`. AASIST detects 40 of the 62 ElevenLabs/2061 clips missed by v3, but a small score sum (`XLS-R logit + 0.1 × AASIST logit`) raises pooled minDCF to 0.0678; grouped out-of-fold logistic stacking is worse (0.2128 pooled). It is not a viable fusion partner on the available evidence.
3. Warm-start XLS-R from v3 and train with class-independent real MP3/Opus/AAC/µ-law round trips, stronger noise coverage, and higher sampling of ElevenLabs/PlayHT. Compare against v3 on the *same* crop and robustness protocols. Keep the original training defaults reproducible.
4. Before claiming open-world generalization, run leave-one-generator-out retraining and obtain a genuinely independent bona fide recording set. The six Grok clips show detection on one new generator, but six clips are not a reliable error-rate estimate. The previous `--exclude-generators` training path still selected checkpoints on validation examples from the excluded generators; `ml/train.py` now masks those examples from checkpoint selection so an eventual unseen-generator result is not selected using its own labels.
5. After choosing a model, regenerate the TSV and validate its score orientation with the organizers. The official scoring code assumes higher = bona fide; the HEARSAY template documentation describes higher = synthetic. An excellent model with inverted submitted scores would fail.

The current validation set has been used for checkpoint selection and calibration. Further tuning on it makes its 0.0656 score increasingly optimistic. Treat new validation gains as candidate-selection evidence; use independent speakers, generators, and recording conditions for a credible final estimate.

The timeline localizes high-scoring windows, but there is no labelled mixed human/synthetic speech benchmark in this workspace. Its region boundaries and live microphone behavior should not be presented as measured localization accuracy. A matched-speaker training set for the **eight training clone identities** would be the most direct way to test and reduce identity shortcuts while preserving speakers 2061 and 6167 as holdouts; any added recordings need clip-level deduplication against the contest test set before training. Identity leakage has been documented as a failure mode in audio deepfake detection ([artifact-centric identity leakage study](https://ieeexplore.ieee.org/abstract/document/11610268)).

The eight training clone IDs are **100, 1487, 3654, 4490, 5448, 6575, 7995, 8848**. Their genuine recordings live in LibriSpeech train-clean-360, which [OpenSLR publishes as a 21 GB archive](https://www.openslr.org/resources/12/). The [Hugging Face dataset viewer filter API](https://huggingface.co/docs/dataset-viewer/filter) offers a route to fetch only those speaker rows from the Parquet copy. The current sandbox refused direct network requests to that API, so no recordings were added and no training result is claimed from them. Keep IDs 2061 and 6167 excluded from this data acquisition to preserve the speaker holdout.

## Completed candidate race

The AASIST run stopped at step 600 after two full validation passes. A warm-start XLS-R candidate (`models/xlsr12_v4_codec_hard`) stopped at step 400 after two passes: full validation minDCF was 0.0854 at step 200 and 0.0799 at step 400, versus v3's 0.0656. The v4 recipe used 50% real codec augmentation, stronger noise, and 50% of spoof draws from ElevenLabs/PlayHT. It worsened the specific ElevenLabs/2061 subgroup (0.2889 versus 0.2415) and PlayHT/2061 (0.1364 versus 0.1071). Simple v3/v4 logit averages also failed to beat v3. The six Grok Voice clips remained flagged, but this sample is too small to distinguish the models.

The fixed 1,400-clip channel suite shows that v4 did not buy robustness:

| Condition | v3 minDCF | v4 minDCF | v3 bona fide FRR at clean threshold | v4 FRR |
|---|---:|---:|---:|---:|
| Clean | 0.0543 | 0.0700 | 2.57% | 3.00% |
| MP3 64 kbps | 0.1229 | 0.1314 | 14.71% | 15.57% |
| Opus 24 kbps | 0.0957 | 0.1171 | 6.14% | 6.71% |
| AAC 48 kbps | 0.0729 | 0.0714 | 5.57% | 5.86% |
| µ-law 8 kHz | 0.1329 | 0.1486 | 9.43% | 14.00% |
| Noise 20 dB | 0.1529 | 0.1714 | 39.86% | 40.43% |
| Noise 10 dB | 0.2471 | 0.2757 | 90.43% | 93.43% |
| Clipping | 0.1271 | 0.1600 | 2.00% | 2.57% |

Full results are in `outputs/results/robustness_xlsr12_v4_codec_hard.json`. v3 stays selected; its TSV was not regenerated. This one failed recipe does not establish that codec augmentation cannot help. It changed codec exposure, noise, and generator sampling together, and the current validation set is small in the hardest strata. The next controlled experiment is matched genuine speech for the eight training clone speakers, with identity-balanced training and a speaker-disjoint validation split. Separately, test codec-only augmentation against a fixed clean/channel holdout, using multiple seeds before choosing a checkpoint. Obtain a new bona fide recording set before treating any validation improvement as a generalization result.
