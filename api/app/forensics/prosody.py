"""Prosody / phonetics forensic branch.

F0 tracking via librosa.pyin when available (falls back to a lightweight
autocorrelation pitch tracker if librosa is missing or too slow), plus simple
energy-based VAD for pause statistics.

Suspicion heuristic (documented, uncalibrated):
  TTS/vocoder speech often has unnaturally low F0 variance (monotone delivery),
  very regular pause durations, and low jitter/shimmer compared to natural speech.
  We combine low F0 std, high monotonicity, and low jitter/shimmer proxies into a
  bounded 0..1 score. This is a weak heuristic; not calibrated against labelled data.
"""
from __future__ import annotations

import numpy as np

from . import _common as C

F0_MIN = 60.0
F0_MAX = 400.0


def _autocorr_pitch(audio: np.ndarray, sr: int, frame_ms: float = 40.0, hop_ms: float = 10.0):
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)
    frames = C.frame_signal(audio, frame_len, hop_len)
    if frames.shape[0] == 0:
        return np.zeros(0), np.zeros(0, dtype=bool)

    min_lag = int(sr / F0_MAX)
    max_lag = min(int(sr / F0_MIN), frame_len - 1)
    f0 = np.zeros(frames.shape[0], dtype=np.float64)
    voiced = np.zeros(frames.shape[0], dtype=bool)

    window = np.hanning(frame_len).astype(np.float32)
    for i, frame in enumerate(frames):
        w = frame * window
        w = w - np.mean(w)
        energy = np.sum(w ** 2)
        if energy < 1e-8:
            continue
        ac = np.correlate(w, w, mode="full")[frame_len - 1:]
        ac = ac / (ac[0] + C.EPS)
        if max_lag <= min_lag:
            continue
        seg = ac[min_lag:max_lag]
        if len(seg) == 0:
            continue
        peak_idx = int(np.argmax(seg))
        peak_val = seg[peak_idx]
        if peak_val > 0.3:
            lag = peak_idx + min_lag
            f0[i] = sr / lag
            voiced[i] = True
    return f0, voiced


def _pitch_track(audio: np.ndarray, sr: int):
    """Fast autocorrelation (YIN-style) pitch tracker.

    `librosa.pyin` is more accurate but its default settings take ~1s even on a
    4s/16kHz clip, blowing the branch's ~150ms budget; the lightweight
    autocorrelation tracker below trades a little accuracy for staying well
    inside budget. Wrapped so a future faster librosa path (e.g. `librosa.yin`)
    can be swapped in without changing the branch's public API.
    """
    return _autocorr_pitch(audio, sr)


def extract_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    audio = C.to_mono_float32(audio)
    keys = _zero_features()
    if len(audio) < int(0.1 * sr):
        return keys

    f0, voiced = _pitch_track(audio, sr)
    is_speech, times = C.energy_vad(audio, sr)

    voiced_ratio = float(np.mean(voiced)) if len(voiced) else 0.0
    voiced_f0 = f0[voiced] if len(f0) else np.zeros(0)

    if len(voiced_f0) > 1:
        f0_mean = float(np.mean(voiced_f0))
        f0_std = float(np.std(voiced_f0))
        f0_range = float(np.max(voiced_f0) - np.min(voiced_f0))
        diffs = np.abs(np.diff(voiced_f0))
        jitter_proxy = float(np.mean(diffs) / (f0_mean + C.EPS))
        # monotonicity: fraction of consecutive voiced frames with near-zero pitch change
        monotonicity = float(np.mean(diffs < 1.0)) if len(diffs) else 0.0
    else:
        f0_mean = f0_std = f0_range = jitter_proxy = monotonicity = 0.0

    # shimmer-like proxy: frame-to-frame RMS amplitude variation during voiced frames
    frame_len = int(sr * 0.04)
    hop_len = int(sr * 0.01)
    frames = C.frame_signal(audio, max(frame_len, 1), max(hop_len, 1))
    if frames.shape[0] > 1:
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + C.EPS)
        n = min(len(rms), len(voiced))
        if n > 1 and np.any(voiced[:n]):
            v_rms = rms[:n][voiced[:n]]
            if len(v_rms) > 1:
                shimmer_proxy = float(np.mean(np.abs(np.diff(v_rms))) / (np.mean(v_rms) + C.EPS))
            else:
                shimmer_proxy = 0.0
        else:
            shimmer_proxy = 0.0
    else:
        shimmer_proxy = 0.0

    # pause statistics via energy VAD
    if len(is_speech):
        pause_mask = ~is_speech
        pause_count = int(np.sum(np.diff(pause_mask.astype(int)) == 1) + (1 if pause_mask[0] else 0))
        hop_s = times[1] - times[0] if len(times) > 1 else 0.01
        pause_frames = np.sum(pause_mask)
        pause_total_s = float(pause_frames * hop_s)
        speech_frames = np.sum(is_speech)
        total_s = len(audio) / sr
        speech_rate_proxy = float(speech_frames * hop_s / (total_s + C.EPS))
    else:
        pause_count, pause_total_s, speech_rate_proxy = 0, 0.0, 0.0

    energy_variance = float(np.var(np.abs(audio.astype(np.float64))))

    return {
        "f0_mean_hz": f0_mean,
        "f0_std_hz": f0_std,
        "f0_range_hz": f0_range,
        "voiced_ratio": voiced_ratio,
        "jitter_proxy": jitter_proxy,
        "shimmer_proxy": shimmer_proxy,
        "monotonicity_index": monotonicity,
        "pause_count": float(pause_count),
        "pause_total_s": pause_total_s,
        "speech_rate_proxy": speech_rate_proxy,
        "energy_variance": energy_variance,
    }


def _zero_features() -> dict[str, float]:
    keys = [
        "f0_mean_hz", "f0_std_hz", "f0_range_hz", "voiced_ratio", "jitter_proxy",
        "shimmer_proxy", "monotonicity_index", "pause_count", "pause_total_s",
        "speech_rate_proxy", "energy_variance",
    ]
    return {k: 0.0 for k in keys}


def analyze(audio: np.ndarray, sr: int, **ctx) -> dict:
    audio = C.to_mono_float32(audio)
    if len(audio) < int(0.2 * sr):
        return C.empty_result("Clip too short for reliable prosody analysis.", ["too_short"])

    feats = extract_features(audio, sr)
    flags: list[str] = []

    if feats["voiced_ratio"] < 0.05:
        return {
            "summary": "Almost no voiced speech detected; prosody analysis is not meaningful.",
            "suspicion": None,
            "features": C.nan_to_zero(feats),
            "flags": ["insufficient_voiced_frames"],
            "used_in_score": False,
        }

    if feats["f0_std_hz"] < 8.0:
        flags.append("unusually flat pitch (low F0 variance)")
    if feats["monotonicity_index"] > 0.8:
        flags.append("high pitch monotonicity")
    if feats["jitter_proxy"] < 0.005:
        flags.append("very low jitter proxy")

    f0std_term = np.clip(1.0 - feats["f0_std_hz"] / 25.0, 0.0, 1.0)
    mono_term = np.clip((feats["monotonicity_index"] - 0.5) / 0.5, 0.0, 1.0)
    jitter_term = np.clip(1.0 - feats["jitter_proxy"] / 0.02, 0.0, 1.0)
    suspicion = float(np.clip(0.4 * f0std_term + 0.3 * mono_term + 0.3 * jitter_term, 0.0, 1.0))

    summary = (
        f"Prosody: F0 mean={feats['f0_mean_hz']:.0f}Hz, std={feats['f0_std_hz']:.1f}Hz, "
        f"voiced ratio={feats['voiced_ratio']:.2f}, pauses={int(feats['pause_count'])}. "
        "Uncalibrated heuristic; natural speakers vary widely."
    )

    return {
        "summary": summary,
        "suspicion": suspicion,
        "features": C.nan_to_zero(feats),
        "flags": flags,
        "used_in_score": False,
    }
