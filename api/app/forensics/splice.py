"""Splice / discontinuity forensic branch.

Frame-wise "seam" scores that combine four independent discontinuity cues between
adjacent short frames:
  1. spectral envelope change (L2 distance between normalized log-magnitude spectra)
  2. noise-floor / RMS jump (dB)
  3. DC-offset jump
  4. phase-derivative discontinuity (mean absolute change in unwrapped phase
     difference between consecutive frames, a coarse proxy for phase splicing)

Suspicion heuristic (documented, uncalibrated):
  A single strong, isolated seam (much larger than the clip's typical frame-to-frame
  variation) is weak evidence of a cut/splice point. Natural speech has continuous
  seams from plosives and breaths too, so this only flags outliers relative to the
  clip's own seam-score distribution (robust z-score), not an absolute threshold.
"""
from __future__ import annotations

import numpy as np

from . import _common as C

FRAME_MS = 32.0
HOP_MS = 16.0
Z_THRESHOLD = 4.0  # robust z-score for flagging a seam as a splice candidate


def _analysis_frames(audio: np.ndarray, sr: int):
    frame_len = max(2, int(sr * FRAME_MS / 1000.0))
    hop_len = max(1, int(sr * HOP_MS / 1000.0))
    n_fft = 1
    while n_fft < frame_len:
        n_fft *= 2
    frames = C.frame_signal(audio, frame_len, hop_len)
    window = np.hanning(frame_len).astype(np.float32)
    windowed = frames * window
    spec = np.fft.rfft(windowed, n=n_fft, axis=1)
    mag = np.abs(spec)
    phase = np.angle(spec)
    return frames, mag, phase, hop_len


def _seam_scores(audio: np.ndarray, sr: int):
    frames, mag, phase, hop_len = _analysis_frames(audio, sr)
    n_frames = frames.shape[0]
    if n_frames < 3:
        return np.zeros(0), hop_len

    # 1. spectral envelope distance
    norm_mag = mag / (np.sum(mag, axis=1, keepdims=True) + C.EPS)
    log_mag = np.log(norm_mag + C.EPS)
    spec_dist = np.sqrt(np.sum(np.diff(log_mag, axis=0) ** 2, axis=1))

    # 2. RMS / noise-floor jump (dB)
    rms_db = C.frame_rms_db(frames)
    rms_jump = np.abs(np.diff(rms_db))

    # 3. DC offset jump
    dc = np.mean(frames.astype(np.float64), axis=1)
    dc_jump = np.abs(np.diff(dc))

    # 4. phase-derivative discontinuity (coarse): mean abs change of phase diff
    phase_diff = np.diff(phase, axis=0)
    phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
    phase_jump = np.mean(np.abs(np.diff(phase_diff, axis=0)), axis=1) if phase_diff.shape[0] > 1 else np.zeros(0)
    # align length: phase_jump has n_frames-2 entries; pad front to match n_frames-1
    if len(phase_jump) < len(spec_dist):
        phase_jump = np.concatenate([[phase_jump[0] if len(phase_jump) else 0.0], phase_jump])

    def _z(x):
        if len(x) == 0:
            return x
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + C.EPS
        return 0.6745 * (x - med) / mad

    z_spec = _z(spec_dist)
    z_rms = _z(rms_jump)
    z_dc = _z(dc_jump)
    z_phase = _z(phase_jump) if len(phase_jump) == len(spec_dist) else np.zeros_like(spec_dist)

    combined = (np.abs(z_spec) + np.abs(z_rms) + np.abs(z_dc) + np.abs(z_phase)) / 4.0
    return combined, hop_len


def extract_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    audio = C.to_mono_float32(audio)
    keys = _zero_features()
    if len(audio) < int(0.1 * sr):
        return keys

    scores, _ = _seam_scores(audio, sr)
    if len(scores) == 0:
        return keys

    return {
        "seam_score_max": float(np.max(scores)),
        "seam_score_mean": float(np.mean(scores)),
        "seam_score_std": float(np.std(scores)),
        "num_candidate_splices": float(np.sum(scores > Z_THRESHOLD)),
    }


def _zero_features() -> dict[str, float]:
    return {
        "seam_score_max": 0.0, "seam_score_mean": 0.0,
        "seam_score_std": 0.0, "num_candidate_splices": 0.0,
    }


def analyze(audio: np.ndarray, sr: int, **ctx) -> dict:
    audio = C.to_mono_float32(audio)
    if len(audio) < int(0.1 * sr):
        return C.empty_result("Clip too short for splice analysis.", ["too_short"])

    scores, hop_len = _seam_scores(audio, sr)
    feats = extract_features(audio, sr)

    if len(scores) == 0:
        return {
            "summary": "Not enough frames to evaluate splice discontinuities.",
            "suspicion": None,
            "features": feats,
            "flags": ["insufficient_frames"],
            "used_in_score": False,
        }

    candidate_idx = np.where(scores > Z_THRESHOLD)[0]
    regions = []
    flags: list[str] = []
    frame_ms_per_hop = hop_len / sr * 1000.0
    for idx in candidate_idx:
        # seam i is the boundary between frame i and frame i+1
        t_ms = (idx + 1) * frame_ms_per_hop
        window_ms = frame_ms_per_hop
        regions.append({
            "start_ms": round(max(t_ms - window_ms, 0.0), 1),
            "end_ms": round(t_ms + window_ms, 1),
            "reason": f"seam score z={scores[idx]:.1f} (spectral/RMS/DC/phase discontinuity)",
        })
    if regions:
        flags.append(f"{len(regions)} candidate splice point(s) detected")

    n_candidates = len(regions)
    if n_candidates == 0:
        suspicion = 0.0
    else:
        # more candidates / stronger max z => higher suspicion, saturating
        strength = np.clip((feats["seam_score_max"] - Z_THRESHOLD) / (2 * Z_THRESHOLD), 0.0, 1.0)
        count_term = np.clip(n_candidates / 5.0, 0.0, 1.0)
        suspicion = float(np.clip(0.7 * strength + 0.3 * count_term, 0.0, 1.0))

    summary = (
        f"Splice/discontinuity scan: {n_candidates} candidate seam(s) above robust "
        f"z-score {Z_THRESHOLD}. Natural speech also has legitimate discontinuities "
        "(plosives, breaths); this flags outliers only, not confirmed edits."
    )

    return {
        "summary": summary,
        "suspicion": suspicion,
        "features": C.nan_to_zero(feats),
        "flags": flags,
        "used_in_score": False,
        "regions": regions,
    }
