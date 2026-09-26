"""Compression / transcoding evidence branch.

Cheap frequency-domain checks for lossy-codec fingerprints: hard band-limiting
(e.g. old MP3 profiles cutting near 16 kHz on 44.1k material, or TTS vocoders
capping near 7-8 kHz on 16k material), spectral "holes" from quantization, and a
double-compression proxy (periodicity in the spectral envelope from repeated
lossy encode/decode cycles).

Suspicion heuristic (documented, uncalibrated):
  A hard band-limit well below Nyquist plus a high fraction of near-zero
  high-band bins is consistent with lossy transcoding or low-bitrate synthesis
  vocoders. This is circumstantial, not proof -- lossless recordings of
  band-limited sources (e.g. telephone audio) look identical.
"""
from __future__ import annotations

import numpy as np

from . import _common as C

FRAME_MS = 32.0
HOP_MS = 16.0


def _spectrum(audio: np.ndarray, sr: int):
    frame_len = max(2, int(sr * FRAME_MS / 1000.0))
    hop_len = max(1, int(sr * HOP_MS / 1000.0))
    n_fft = 1
    while n_fft < frame_len:
        n_fft *= 2
    window = np.hanning(frame_len).astype(np.float32)
    frames = C.frame_signal(audio, frame_len, hop_len)
    mag = C.rfft_magnitude(frames, n_fft, window)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    return mag, freqs


def extract_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    audio = C.to_mono_float32(audio)
    keys = _zero_features()
    if len(audio) < int(0.05 * sr):
        return keys

    mag, freqs = _spectrum(audio, sr)
    if mag.shape[0] == 0:
        return keys
    power = mag ** 2
    med_spectrum = np.median(power, axis=0)
    nyq = sr / 2.0

    total_med_energy = float(np.sum(med_spectrum)) + C.EPS
    cumsum_med = np.cumsum(med_spectrum)
    cutoff_bin = int(np.argmax(cumsum_med >= 0.999 * total_med_energy))
    cutoff_hz = float(freqs[cutoff_bin])

    # fraction of near-zero bins in the top quarter of the spectrum (quantization holes)
    hi_start = int(len(freqs) * 0.75)
    hi_band = power[:, hi_start:]
    hi_floor = np.percentile(power, 5)
    near_zero_frac = float(np.mean(hi_band < (hi_floor * 2.0 + C.EPS)))

    # "shelf" detection: energy ratio just below vs just above the estimated cutoff
    shelf_width_hz = 500.0
    below_mask = (freqs >= max(cutoff_hz - shelf_width_hz, 0)) & (freqs < cutoff_hz)
    above_mask = (freqs >= cutoff_hz) & (freqs < min(cutoff_hz + shelf_width_hz, nyq))
    below_energy = float(np.sum(med_spectrum[below_mask])) + C.EPS
    above_energy = float(np.sum(med_spectrum[above_mask])) + C.EPS
    shelf_ratio = float(above_energy / below_energy)

    # double-compression proxy: periodicity of the spectral envelope's second derivative
    # (block-DCT quantization from repeated lossy encodes tends to leave a periodic comb).
    log_env = np.log(med_spectrum + C.EPS)
    d2 = np.diff(log_env, n=2)
    if len(d2) > 8:
        spec_of_env = np.abs(np.fft.rfft(d2 - np.mean(d2)))
        total = np.sum(spec_of_env) + C.EPS
        peak = np.max(spec_of_env[1:]) if len(spec_of_env) > 1 else 0.0
        double_compression_proxy = float(peak / total)
    else:
        double_compression_proxy = 0.0

    return {
        "band_limit_cutoff_hz": cutoff_hz,
        "cutoff_ratio_of_nyquist": float(cutoff_hz / max(nyq, 1.0)),
        "highband_near_zero_fraction": near_zero_frac,
        "shelf_ratio": shelf_ratio,
        "double_compression_proxy": double_compression_proxy,
    }


def _zero_features() -> dict[str, float]:
    keys = [
        "band_limit_cutoff_hz", "cutoff_ratio_of_nyquist", "highband_near_zero_fraction",
        "shelf_ratio", "double_compression_proxy",
    ]
    return {k: 0.0 for k in keys}


def analyze(audio: np.ndarray, sr: int, **ctx) -> dict:
    audio = C.to_mono_float32(audio)
    if len(audio) < int(0.05 * sr):
        return C.empty_result("Clip too short for compression analysis.", ["too_short"])

    feats = extract_features(audio, sr)
    flags: list[str] = []

    if feats["cutoff_ratio_of_nyquist"] < 0.9:
        flags.append(f"band-limited to ~{feats['band_limit_cutoff_hz']:.0f} Hz "
                      f"({feats['cutoff_ratio_of_nyquist']*100:.0f}% of Nyquist)")
    if feats["highband_near_zero_fraction"] > 0.6:
        flags.append("large fraction of near-zero high-band energy (possible quantization)")
    if feats["shelf_ratio"] < 0.1:
        flags.append("sharp spectral shelf near band-limit cutoff")

    cutoff_term = np.clip(1.0 - feats["cutoff_ratio_of_nyquist"], 0.0, 1.0)
    shelf_term = np.clip(1.0 - feats["shelf_ratio"] / 0.3, 0.0, 1.0)
    suspicion = float(np.clip(0.6 * cutoff_term + 0.4 * shelf_term, 0.0, 1.0))

    summary = (
        f"Compression evidence: band-limit ~{feats['band_limit_cutoff_hz']:.0f}Hz "
        f"({feats['cutoff_ratio_of_nyquist']*100:.0f}% of Nyquist), "
        f"shelf ratio={feats['shelf_ratio']:.2f}. "
        "Consistent with lossy transcoding or a band-limited vocoder, but not proof of either."
    )

    return {
        "summary": summary,
        "suspicion": suspicion,
        "features": C.nan_to_zero(feats),
        "flags": flags,
        "used_in_score": False,
    }
