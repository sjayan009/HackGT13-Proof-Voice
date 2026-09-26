"""Shared helpers for forensic branches.

Kept dependency-light (numpy only) so every branch stays fast and deterministic.
All functions are pure / side-effect free and operate on float32 mono audio.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-10


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Coerce input to 1-D float32 mono in roughly [-1, 1]."""
    arr = np.asarray(audio)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    arr = arr.astype(np.float32, copy=False)
    return np.ascontiguousarray(arr)


def frame_signal(audio: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """Return a (n_frames, frame_len) view/array of overlapping frames.

    Pads the tail with zeros so every sample is covered by at least one frame.
    Deterministic, no randomness.
    """
    n = len(audio)
    if n == 0:
        return np.zeros((0, frame_len), dtype=np.float32)
    if n < frame_len:
        padded = np.zeros(frame_len, dtype=np.float32)
        padded[:n] = audio
        return padded.reshape(1, frame_len)
    n_frames = 1 + (n - frame_len) // hop_len
    # ensure we cover the tail
    if (n_frames - 1) * hop_len + frame_len < n:
        n_frames += 1
    pad_needed = (n_frames - 1) * hop_len + frame_len - n
    if pad_needed > 0:
        audio = np.concatenate([audio, np.zeros(pad_needed, dtype=np.float32)])
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, frame_len),
        strides=(audio.strides[0] * hop_len, audio.strides[0]),
        writeable=False,
    )
    return frames.copy()


def frame_rms_db(frames: np.ndarray) -> np.ndarray:
    """RMS in dBFS per frame (silence floored at -120 dB)."""
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + EPS)
    return 20.0 * np.log10(np.maximum(rms, 1e-6))


def energy_vad(audio: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0,
                threshold_db: float = -40.0) -> tuple[np.ndarray, np.ndarray]:
    """Simple energy-based voice-activity detector.

    Returns (frame_is_speech: bool array, frame_times_s: float array of frame start times).
    threshold_db is relative to the clip's own noise floor estimate (10th percentile RMS),
    not an absolute level, so it works across recording conditions.
    """
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop_len = max(1, int(sr * hop_ms / 1000.0))
    frames = frame_signal(audio, frame_len, hop_len)
    if frames.shape[0] == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)
    rms_db = frame_rms_db(frames)
    noise_floor = np.percentile(rms_db, 10)
    is_speech = rms_db > (noise_floor + abs(threshold_db) * 0 + 12.0)
    # 12 dB above the estimated noise floor => speech-like frame
    times = np.arange(frames.shape[0], dtype=np.float64) * hop_len / sr
    return is_speech, times


def safe_float(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(v):
        return 0.0
    return v


def nan_to_zero(d: dict) -> dict:
    return {k: safe_float(v) for k, v in d.items()}


def rfft_magnitude(frames: np.ndarray, n_fft: int, window: np.ndarray | None = None) -> np.ndarray:
    """Magnitude spectrum per frame using rfft. frames: (n_frames, frame_len)."""
    if window is not None:
        frames = frames * window
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    return np.abs(spec)


def empty_result(summary: str, flags: list[str] | None = None) -> dict:
    return {
        "summary": summary,
        "suspicion": None,
        "features": {},
        "flags": flags or [],
        "used_in_score": False,
    }
