# Telephony call-level evaluation (simulated)

Frozen, speaker-disjoint call set + simulated channels + per-hop traces of the serving model.
**Everything here is simulated from the HEARSAY validation set (read LibriSpeech/LJ speech and TTS clips).
It is not real telephone data and must not be quoted as production accuracy.**

## Pipeline

```bash
python ml/telephony_eval/build_calls.py          # outputs/telephony_eval/calls.json (manifest only, seed 20261124)
python ml/telephony_eval/channels.py             # self-check of the channel simulations
python ml/telephony_eval/score_traces.py         # traces for models/selected, all calls, 7 channels (~40 min GPU)
python ml/telephony_eval/score_traces.py --model-dir models/xlsr12_v4_codec_hard --split eval \
    --out-dir outputs/telephony_eval/xlsr12_v4_codec_hard
python ml/telephony_eval/score_traces.py --split eval --channels clean telephone --no-floor \
    --out-dir outputs/telephony_eval/ablation_nofloor    # noise-floor ablation
python ml/telephony_eval/analyze.py --traces outputs/telephony_eval outputs/telephony_eval/xlsr12_v4_codec_hard \
    outputs/telephony_eval/ablation_nofloor --write-calibration                          # outputs/results/telephony_eval.json, models/selected/channel_calibration.json
```

## Calls (`outputs/telephony_eval/calls.json`)

514 calls, 20-40 s each (4.8 h of audio). Validation clips are concatenated with 0.3-1.0 s pauses;
each clip is scaled to -26 dBFS RMS (peak-limited, never clipped) and a continuous -70 dBFS white floor is
added so silence is never digital zero. No clip is used twice.

| split | genuine | synthetic | mixed | voices |
|---|---:|---:|---:|---|
| calib | 100 | 120 | 24 | 8 LibriSpeech speakers; clones of speaker_6167 (6 generators) |
| eval | 112 | 130 | 28 | the other 8 LibriSpeech speakers + LJ; clones of speaker_2061 (6 generators) + LJ-voice (4 generators) |

* `mixed` = genuine speech, then synthetic speech from `switch_ms` (start of the first synthetic clip).
  LJ mixed calls switch to the LJ voice (same claimed speaker); LibriSpeech mixed calls switch to the split's
  cloned voice (a different voice, so these are easier than a real takeover).
* The split is speaker-disjoint: calibration/gates fitted on `calib` never see an `eval` voice. The documented
  hard cloned speaker (2061) is in eval; calib therefore contains only the easier clone and no single-speaker
  (LJ-voice) generators. Calls from one speaker are correlated; CIs bootstrap over calls, not speakers.
* Each call entry: `call_id, split, kind, speaker, generator, seed, duration_ms, switch_ms, segments[]`,
  segment = `{val_idx (row of outputs/manifests/val.csv), start_ms (in call), dur_ms, y}`.

## Channels (`channels.py`, deterministic per call seed)

| channel | simulation |
|---|---|
| `clean` | rendered call, 16 kHz wideband |
| `telephone` | soxr HQ 16k->8k, `api/app/telephony/mulaw.py` encode/decode, soxr HQ 8k->16k (the live path) |
| `telephone_noise30` / `telephone_noise20` | pink noise 30 / 20 dB below the -26 dBFS speech level, added before the phone channel |
| `telephone_loss5` | telephone with 5 % of 20 ms packets zero-filled, Gilbert-Elliott bursts (mean 3 packets) |
| `opus_telephone` | Opus 24 kbps round trip (ffmpeg/libopus, as `ml/robustness.py`), then telephone |
| `replay_sim` | crude replay proxy: random exponential-decay room IR (RT60 0.25-0.6 s) + 150-6000 Hz band-pass; wideband. Not real loudspeaker/microphone replay |

Channel families for calibration: `clean` is fitted on `clean` only (`replay_sim` is evaluated with the clean
calibration but never used to fit it); `telephone` is fitted on the five telephone channels pooled.

## Trace schema (`outputs/telephony_eval/traces_<channel>.npz`)

One file per channel; one row per scored window. Load with `np.load(path)`.
The selected model's traces cover all 514 calls (both splits); other models' traces live in
`outputs/telephony_eval/<model dir name>/` and usually cover `eval` only.

Live-system replay semantics: 16 kHz audio, 0.5 s hops. For hop end `t` = 0.5, 1.0, ... s the `FloorTracker`
is updated with the newest 0.5 s hop; once `t >= 3.0 s` the window `audio[t-3 s, t]` is scored with
`PrimaryDetector.logits` (FP16, serving code path) and `window_quality(window, floor_db=FloorTracker.floor)`.
No window is scored before 3.0 s; a trailing partial hop is dropped. Telephone audio is resampled for the whole
call at once (equivalent to a streaming soxr resampler up to edge effects).

Per-window arrays (length = number of windows, all calls concatenated, sorted by call then time):

| key | dtype | meaning |
|---|---|---|
| `call_index` | int32 | index into the per-call arrays below |
| `t_ms` | int32 | window END time in the call (window = `[t_ms-3000, t_ms)`) |
| `logit` | float32 | raw model logit (>0 = synthetic). Serving probability: `sigmoid(a*logit+b)` with `meta.calibration` |
| `y` | int8 | 0 genuine, 1 synthetic, -1 mixed-call window straddling the switch |
| `speech_s` | float32 | `window_quality` speech seconds (energy VAD, call-level floor) |
| `snr_db` | float32 | `window_quality` SNR (median speech frame level - floor) |
| `level_dbfs`, `clip_ratio`, `floor_db` | float32 | `window_quality` fields |
| `loss_frac` | float32 | fraction of 20 ms packets in the window that were lost (0 except `telephone_loss5`) |
| `hop_speech_s` | float32 | `window_quality(audio[t-0.5 s, t], floor_db=FloorTracker.floor)["speech_s"]`: speech in the NEWEST hop, same floor as the window (Aegis gate input) |

Per-call arrays (length = number of calls): `call_id, call_split, call_kind, call_speaker, call_generator`
(str), `call_duration_ms, call_switch_ms` (int32, -1 = no switch), `call_loss_frac` (float32).
Scalars: `channel` (str), `meta` (JSON str: model_dir, model_version, calibration, calls_seed, split, win/hop).

```python
import numpy as np, json
t = np.load("outputs/telephony_eval/traces_telephone.npz")
cal = json.loads(str(t["meta"]))["calibration"]
p = 1 / (1 + np.exp(-(cal["a"] * t["logit"] + cal["b"])))
for c in np.flatnonzero(t["call_split"] == "eval"):
    w = t["call_index"] == c          # this call's hops, in time order
    ...                               # replay a risk policy over t["t_ms"][w], p[w], t["speech_s"][w]
```

## Quality gates (recommendation)

Gates were chosen on the `calib` half, then checked on `eval` (`outputs/results/telephony_eval.json` ->
`abstention`, `models.selected.gated`). Selected model, serving threshold, window level:

| gate | why | effect (eval) |
|---|---|---|
| `speech_s >= 1.0` (window) | sanity gate against near-silent windows | costs <= 2 % coverage; no measurable error change here because the simulated calls are dense speech |
| `snr_db >= 20` | the only gate with a real effect | telephone_noise20: genuine FA 36.2 % -> 27.1 % at 73 % coverage; telephone family overall 26.2 % -> 23.8 % at 95 % |
| `loss_frac <= 0.10` | drop windows with burst loss | telephone_loss5: FA 33.4 % -> 30.9 % at 90 % coverage |

Default combined gate in `analyze.py`: `speech_s>=1.0,snr_db>=20,loss_frac<=0.1`.

What the gates do NOT fix: the dominant false alarms come from the telephone channel itself on clean,
high-SNR speech (telephone FA 20 % with median SNR 38 dB) and from particular voices (LJ: 45-81 % window FA
depending on channel). Quality gating removes at most about a quarter of the false alarms, and only in heavy
noise. It is not a substitute for telephony-aware training or call-level persistence (backend policy).

Backend issues found:
* **Packet loss breaks the noise floor.** Zero-filled packets drive `FloorTracker` to about -123 dB (median), so
  `snr_db` reads about 94 dB and an SNR gate passes everything under loss. Exclude lost/zero packets from
  `FloorTracker.update` (or insert comfort noise/PLC before VAD). Not changed here (`api/` is not ours).
* **The detector reacts to a faint noise floor.** Removing the -70 dBFS white floor (44 dB below speech) halves
  clean genuine FA (12.1 % -> 6.1 %, `selected_nofloor_ablation`). Telephone is barely affected (20.2 % -> 18.3 %).
  Real calls always have a floor, so the floor stays in the main protocol.
* For the Aegis gate: `hop_speech_s < 0.1 s` covers 11-16 % of hops per channel. That is the compute an
  "is anyone speaking" skip could save on these dense calls; real calls with longer silences will save more.
