"""Spectral forensic branch.

Computes log-mel / spectral-shape statistics that are cheap (< ~150 ms for a 4 s
16 kHz clip) and CPU-only. Target audio is 16 kHz (Nyquist 8 kHz); all features are
built with that ceiling in mind (no attempt to look above 8 kHz).

Suspicion heuristic (documented, uncalibrated):
  Many neural TTS/vocoder pipelines produce unnaturally *flat* spectra (low variance
  spectral flux, high spectral flatness) and/or a hard band-limit near 7.5-8 kHz from
  upsampling a lower-rate model output. We combine three weakly-indicative signals
  (high mean flatness, low flux variance, low high-band energy ratio) into a bounded
  0..1 score. This is a heuristic, not a calibrated probability; the ML lead should
  measure its standalone AUC before deciding whether it enters the fused score.
"""
from __future__ import annotations

import numpy as np

from . import _common as C

FRAME_MS = 32.0
HOP_MS = 16.0
N_MEL = 40
FMIN = 20.0


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    bin_pts = np.clip(bin_pts, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left, center, right = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for k in range(left, min(center, fb.shape[1])):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, min(right, fb.shape[1])):
            fb[m - 1, k] = (right - k) / max(right - center, 1)
    return fb


def _spectral_frames(audio: np.ndarray, sr: int):
    frame_len = max(2, int(sr * FRAME_MS / 1000.0))
    hop_len = max(1, int(sr * HOP_MS / 1000.0))
    n_fft = 1
    while n_fft < frame_len:
        n_fft *= 2
    window = np.hanning(frame_len).astype(np.float32)
    frames = C.frame_signal(audio, frame_len, hop_len)
    mag = C.rfft_magnitude(frames, n_fft, window)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    return mag, freqs, hop_len, n_fft


def extract_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    audio = C.to_mono_float32(audio)
    if len(audio) < 4:
        return _zero_features()

    mag, freqs, hop_len, n_fft = _spectral_frames(audio, sr)
    if mag.shape[0] == 0:
        return _zero_features()
    power = mag ** 2 + C.EPS

    # log-mel stats
    nyq = sr / 2.0
    fmax = min(nyq - 1.0, 7900.0)
    try:
        fb = _mel_filterbank(sr, n_fft, N_MEL, FMIN, fmax)
        mel_energy = power @ fb.T  # (n_frames, n_mels)
        log_mel = np.log(mel_energy + C.EPS)
        mel_mean = float(np.mean(log_mel))
        mel_std = float(np.std(log_mel))
    except Exception:
        mel_mean, mel_std = 0.0, 0.0

    # spectral flatness (geometric mean / arithmetic mean of power spectrum)
    log_power = np.log(power)
    geo_mean = np.exp(np.mean(log_power, axis=1))
    arith_mean = np.mean(power, axis=1)
    flatness = geo_mean / np.maximum(arith_mean, C.EPS)
    flatness_mean = float(np.mean(flatness))
    flatness_std = float(np.std(flatness))

    # spectral centroid & rolloff (85%)
    total_energy = np.sum(power, axis=1) + C.EPS
    centroid = np.sum(power * freqs[None, :], axis=1) / total_energy
    centroid_mean = float(np.mean(centroid))

    cumsum = np.cumsum(power, axis=1)
    rolloff_thresh = 0.85 * total_energy
    rolloff_bins = np.argmax(cumsum >= rolloff_thresh[:, None], axis=1)
    rolloff_freqs = freqs[rolloff_bins]
    rolloff_mean = float(np.mean(rolloff_freqs))

    # high-band (4-8kHz) energy ratio
    hi_mask = (freqs >= 4000.0) & (freqs <= min(8000.0, nyq))
    lo_mask = freqs < 4000.0
    hi_energy = np.sum(power[:, hi_mask], axis=1)
    lo_energy = np.sum(power[:, lo_mask], axis=1)
    hi_ratio = hi_energy / (hi_energy + lo_energy + C.EPS)
    hi_ratio_mean = float(np.mean(hi_ratio))

    # band-limit cutoff estimate: lowest frequency below which 99.9% of the median
    # spectrum's energy is contained (robust to windowing leakage, unlike a raw
    # per-bin noise-floor comparison).
    med_spectrum = np.median(power, axis=0)
    total_med_energy = float(np.sum(med_spectrum)) + C.EPS
    cumsum_med = np.cumsum(med_spectrum)
    cutoff_bin = int(np.argmax(cumsum_med >= 0.999 * total_med_energy))
    cutoff_hz = float(freqs[cutoff_bin])

    # spectral flux variance (frame-to-frame change in normalized spectrum)
    norm_mag = mag / (np.sum(mag, axis=1, keepdims=True) + C.EPS)
    if norm_mag.shape[0] > 1:
        flux = np.sqrt(np.sum(np.diff(norm_mag, axis=0) ** 2, axis=1))
        flux_var = float(np.var(flux))
        flux_mean = float(np.mean(flux))
    else:
        flux_var, flux_mean = 0.0, 0.0

    # harmonic-to-noise proxy: ratio of peak energy in low-order harmonics vs total (cheap, no pitch tracking)
    # use spectral peak prominence in 80-1000Hz band as a coarse voiced-harmonic proxy
    voice_mask = (freqs >= 80.0) & (freqs <= 1000.0)
    if np.any(voice_mask):
        band = power[:, voice_mask]
        peak = np.max(band, axis=1)
        mean_band = np.mean(band, axis=1) + C.EPS
        hnr_proxy = float(np.mean(np.log(peak / mean_band + 1.0)))
    else:
        hnr_proxy = 0.0

    return {
        "logmel_mean": mel_mean,
        "logmel_std": mel_std,
        "flatness_mean": flatness_mean,
        "flatness_std": flatness_std,
        "centroid_mean_hz": centroid_mean,
        "rolloff_mean_hz": rolloff_mean,
        "highband_energy_ratio": hi_ratio_mean,
        "band_limit_cutoff_hz": cutoff_hz,
        "flux_variance": flux_var,
        "flux_mean": flux_mean,
        "hnr_proxy": hnr_proxy,
    }


def _zero_features() -> dict[str, float]:
    keys = [
        "logmel_mean", "logmel_std", "flatness_mean", "flatness_std",
        "centroid_mean_hz", "rolloff_mean_hz", "highband_energy_ratio",
        "band_limit_cutoff_hz", "flux_variance", "flux_mean", "hnr_proxy",
    ]
    return {k: 0.0 for k in keys}


def analyze(audio: np.ndarray, sr: int, **ctx) -> dict:
    audio = C.to_mono_float32(audio)
    if len(audio) < int(0.05 * sr):
        return C.empty_result("Clip too short for reliable spectral analysis.", ["too_short"])

    feats = extract_features(audio, sr)
    flags: list[str] = []

    if feats["band_limit_cutoff_hz"] < 7000.0:
        flags.append(f"band-limited near {feats['band_limit_cutoff_hz']:.0f} Hz")
    if feats["flatness_mean"] > 0.35:
        flags.append("unusually flat spectrum (low spectral contrast)")
    if feats["flux_variance"] < 1e-5:
        flags.append("very low frame-to-frame spectral variation")
    if feats["highband_energy_ratio"] < 0.02:
        flags.append("almost no high-band (4-8kHz) energy")

    # bounded heuristic suspicion, see module docstring
    flat_term = np.clip((feats["flatness_mean"] - 0.15) / 0.35, 0.0, 1.0)
    flux_term = np.clip(1.0 - feats["flux_variance"] / 5e-4, 0.0, 1.0)
    hiband_term = np.clip(1.0 - feats["highband_energy_ratio"] / 0.10, 0.0, 1.0)
    suspicion = float(np.clip(0.4 * flat_term + 0.3 * flux_term + 0.3 * hiband_term, 0.0, 1.0))

    summary = (
        f"Spectral shape: flatness={feats['flatness_mean']:.3f}, "
        f"band-limit~{feats['band_limit_cutoff_hz']:.0f}Hz, "
        f"high-band energy ratio={feats['highband_energy_ratio']:.3f}. "
        "This is a weak, uncalibrated heuristic, not proof of synthesis."
    )

    return {
        "summary": summary,
        "suspicion": suspicion,
        "features": C.nan_to_zero(feats),
        "flags": flags,
        "used_in_score": False,
    }
