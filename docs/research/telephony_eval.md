# Telephony evaluation of the ProofVoice detector (simulated calls)

**Status: measured on simulated calls and simulated channels built from the HEARSAY validation set. None of
this is real telephone data. These numbers are not production accuracy and do not show that any PRD target
has been met.** Code and trace schema: `ml/telephony_eval/README.md`. Raw results:
`outputs/results/telephony_eval.json`.

## Method

* **Calls.** 514 calls of 20-40 s (4.8 h), built by concatenating validation clips of one voice with pauses:
  212 genuine, 250 synthetic, 52 mixed (genuine, then synthetic from a recorded switch time). Clips are
  level-normalised to -26 dBFS, a -70 dBFS white floor is added, and no clip is reused.
* **Split.** Speaker-disjoint: `calib` = 8 LibriSpeech speakers + speaker_6167 clones; `eval` = the other 8
  LibriSpeech speakers + LJ + speaker_2061 clones + LJ-voice TTS. Everything fitted (calibration, gates) uses
  calib; everything reported uses eval. Calib therefore has only the easier cloned voice and no LJ-voice
  generators.
* **Channels.** clean; telephone (soxr 16k->8k, the production G.711 μ-law codec, back to 16k); telephone + pink
  noise at 30/20 dB SNR; telephone + 5 % bursty packet loss; Opus 24k -> telephone; and `replay_sim`, a crude
  room-IR + band-pass proxy that is **not** real replay.
* **Scoring.** The live window: 3 s windows every 0.5 s through `PrimaryDetector.logits` (FP16), with
  `window_quality` + `FloorTracker` from `api/app/audio/realtime.py`. Window labels come from the call (mixed
  windows that straddle the switch are excluded). Metrics: EER, AUC, minDCF (HEARSAY costs), and genuine
  false-alarm (FA) / spoof miss at the serving threshold (`calibration.json`, fitted on clean validation clips).
  95 % CIs are a 1,000x call bootstrap stratified by call kind. Calls cluster by speaker, so these CIs
  understate speaker variance; the per-speaker FA spread is reported separately.

## Measured results (eval half, window level)

| channel | v3 EER [95 % CI] | v3 genuine FA @ serving thr | v3 spoof miss | v4 EER | v4 genuine FA |
|---|---|---|---|---|---|
| clean | 7.1 % [5.5, 8.9] | 12.1 % [9.6, 14.7] | 4.9 % | 8.3 % | 13.0 % |
| telephone | 10.5 % [8.1, 12.7] | 20.2 % [16.9, 23.7] | 6.3 % | 10.9 % | 24.1 % |
| telephone_noise30 | 13.5 % [11.1, 15.8] | 18.9 % [16.3, 21.8] | 11.0 % | 14.2 % | 22.4 % |
| telephone_noise20 | 16.8 % [14.5, 19.3] | 36.1 % [31.9, 40.5] | 7.4 % | 17.4 % | 40.6 % |
| telephone_loss5 | 10.4 % [8.1, 12.7] | 33.3 % [29.5, 37.5] | 3.2 % | 10.5 % | 37.0 % |
| opus_telephone | 9.9 % [7.7, 12.0] | 21.9 % [18.3, 25.4] | 5.4 % | 10.1 % | 25.3 % |
| replay_sim | 35.8 % [33.1, 38.5] | 9.6 % | 67.3 % | 36.5 % | 12.9 % |

* Window-level clean EER (7.1 %) is far worse than the 1.5 % clip-level validation EER. Three contributors:
  (1) the hard eval voices (speaker_2061/ElevenLabs miss 33 %; LJ genuine FA 45 %); (2) the faint noise floor:
  removing it gives clean EER 5.2 % and FA 6.1 % (ablation); (3) windows that span clip joins and pauses.
  The same model on calib voices: clean EER 1.4 %, FA 5.3 %.
* Genuine FA varies widely by speaker: 0.1-45 % on clean and 2-67 % on telephone, with LJ worst.
* v4 (codec + noise + hard-generator mix) is not better than v3 on any simulated channel. This agrees with
  the earlier clip-level robustness finding in MODEL_STRATEGY.md.
* Replay proxy: the detector misses about two thirds of synthetic windows. The proxy is crude, but this is the
  largest gap measured.
* Illustrative call-level numbers (the backend owns the real policy): with "3 consecutive windows above the
  serving threshold", 49 % of clean and 58 % of telephone genuine eval calls would alarm. Every synthetic call
  is caught on the telephone channels; 79 % are caught on replay_sim. A per-window threshold with no
  persistence or evidence policy is unusable on calls.

## Channel-conditioned calibration (Workstream B)

Fitted on calib (speech_s >= 1 s): class-balanced Platt per family plus the minDCF threshold. Written to
`models/selected/channel_calibration.json` (clean: a 1.780, b 7.523, thr 0.252; telephone: a 1.162, b 4.369,
thr 0.209; each family also has `threshold_window_FA_1pct`).

| eval channel | balanced ECE: serving -> temperature -> channel Platt | genuine FA / miss: serving thr -> channel thr |
|---|---|---|
| clean | 0.029 -> 0.035 -> 0.040 | 12.1 / 4.9 % -> 5.2 / 8.3 % |
| telephone | 0.058 -> 0.044 -> 0.041 | 20.2 / 6.3 % -> 11.5 / 10.0 % |
| telephone_noise20 | 0.119 -> 0.100 -> 0.076 | 36.1 / 7.4 % -> 25.0 / 11.9 % |
| telephone_loss5 | 0.108 -> 0.115 -> 0.052 | 33.3 / 3.2 % -> 23.1 / 5.9 % |
| opus_telephone | 0.064 -> 0.058 -> 0.036 | 21.9 / 5.4 % -> 13.3 / 8.2 % |

Answer to "can reliability improve without retraining?": **partly.**
* Telephone-family Platt lowers ECE on every telephone channel.
* The channel threshold roughly halves genuine FA, but misses go up. Under the HEARSAY cost, which weights a
  miss four times a false alarm, actDCF gets worse (telephone 0.45 -> 0.51).
* The calib-fitted threshold transfers imperfectly because the calib spoofs come from the easier cloned voice.
* The 1 %-FA thresholds reach 1.2 % window FA on telephone, but miss 21 % of synthetic windows.
* Temperature scaling alone barely changes ECE.
* Ranking metrics (EER, minDCF) cannot change, because the calibration is monotone. Recovering separability
  requires retraining.

## Selective classification / quality gates

Risk-coverage curves for `speech_s`, `snr_db` and `loss_frac` are in `abstention`. The only gate with an effect
is SNR >= 20 dB, which takes noise20 FA from 36 % to 27 % at 73 % coverage. A speech gate does nothing on these
dense calls, and a loss gate is minor. Quality abstention cannot address the main failure: false alarms on
clean, high-SNR telephone speech. Recommended gates and two backend issues (packet loss collapses the
FloorTracker floor; sensitivity to a faint noise floor) are in `ml/telephony_eval/README.md`.

## Targets vs. measured (PRD §11 targets, not achieved)

| metric | target | measured here (simulated, window level, v3) |
|---|---|---|
| telephony EER | <= 3 % | 10.5 % (telephone), 9.9-16.8 % across telephone channels |
| genuine-call false alert | <= 1 % | not measured at call level here (backend policy); window FA 20 % |
| synthetic-call detection | >= 90 % | 100 % (3 consecutive windows) on telephone channels, with an unusable genuine alarm rate |
| controlled evaluation calls | >= 300 | 514 simulated calls (270 eval); 0 real calls |

## Next steps

* The Workstream A queue (`python ml/experiments/queue.py`) is prepared but **not run**: codec-only x2 seeds,
  a 4-layer distilled student, and leave-one-generator-out for ElevenLabs and PlayHT.
* Real telephone recordings of consenting speakers are required before any number here is quoted externally.
