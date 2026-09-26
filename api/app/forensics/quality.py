"""Signal-quality branch: SNR, clipping, silence, level, speech ratio.

This branch is not primarily about deepfake evidence -- it feeds the orchestrator's
`analysis_confidence` and routing decisions (e.g. skip prosody if speech_ratio is
near zero). `suspicion` is always None here: signal quality alone is not evidence
of synthesis.
"""
from __future__ import annotations

import numpy as np

from . import _common as C

CLIP_THRESHOLD = 0.99


def extract_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    audio = C.to_mono_float32(audio)
    keys = _zero_features()
    if len(audio) == 0:
        return keys

    abs_audio = np.abs(audio.astype(np.float64))
    peak = float(np.max(abs_audio)) if len(abs_audio) else 0.0
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + C.EPS))
    rms_dbfs = float(20.0 * np.log10(max(rms, 1e-8)))

    clip_ratio = float(np.mean(abs_audio >= CLIP_THRESHOLD))

    is_speech, times = C.energy_vad(audio, sr)
    speech_ratio = float(np.mean(is_speech)) if len(is_speech) else 0.0
    silence_ratio = float(1.0 - speech_ratio) if len(is_speech) else 1.0

    # SNR estimate: speech-frame RMS power vs non-speech (noise) frame RMS power
    frame_len = int(sr * 0.025)
    hop_len = int(sr * 0.01)
    frames = C.frame_signal(audio, max(frame_len, 1), max(hop_len, 1))
    if frames.shape[0] > 0:
        frame_power = np.mean(frames.astype(np.float64) ** 2, axis=1)
        n = min(len(frame_power), len(is_speech))
        if n > 0 and np.any(is_speech[:n]) and np.any(~is_speech[:n]):
            speech_power = np.mean(frame_power[:n][is_speech[:n]]) + C.EPS
            noise_power = np.mean(frame_power[:n][~is_speech[:n]]) + C.EPS
            snr_db = float(10.0 * np.log10(speech_power / noise_power))
        else:
            snr_db = 0.0
    else:
        snr_db = 0.0

    duration_s = float(len(audio) / sr)

    return {
        "snr_estimate_db": snr_db,
        "clipping_ratio": clip_ratio,
        "silence_ratio": silence_ratio,
        "speech_ratio": speech_ratio,
        "rms_dbfs": rms_dbfs,
        "peak_amplitude": peak,
        "duration_s": duration_s,
    }


def _zero_features() -> dict[str, float]:
    return {
        "snr_estimate_db": 0.0, "clipping_ratio": 0.0, "silence_ratio": 1.0,
        "speech_ratio": 0.0, "rms_dbfs": -120.0, "peak_amplitude": 0.0, "duration_s": 0.0,
    }


def analyze(audio: np.ndarray, sr: int, **ctx) -> dict:
    audio = C.to_mono_float32(audio)
    if len(audio) == 0:
        return C.empty_result("Empty audio; cannot assess quality.", ["empty_audio"])

    feats = extract_features(audio, sr)
    flags: list[str] = []

    if feats["clipping_ratio"] > 0.001:
        flags.append(f"clipping detected ({feats['clipping_ratio']*100:.2f}% of samples)")
    if feats["silence_ratio"] > 0.9:
        flags.append("mostly silence")
    if feats["snr_estimate_db"] < 6.0 and feats["speech_ratio"] > 0.05:
        flags.append("low estimated SNR")
    if feats["rms_dbfs"] < -45.0:
        flags.append("very low signal level")

    summary = (
        f"Signal quality: SNR~{feats['snr_estimate_db']:.1f}dB, "
        f"speech ratio={feats['speech_ratio']:.2f}, clipping={feats['clipping_ratio']*100:.2f}%, "
        f"level={feats['rms_dbfs']:.1f} dBFS. Used for routing/confidence, not synthesis evidence."
    )

    return {
        "summary": summary,
        "suspicion": None,
        "features": C.nan_to_zero(feats),
        "flags": flags,
        "used_in_score": False,
    }
