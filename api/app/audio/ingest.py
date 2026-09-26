"""Audio ingestion: decode any common format to mono float32 at 16 kHz.

soundfile (libsndfile: WAV/FLAC/OGG/MP3) first; ffmpeg (system or imageio-ffmpeg) for everything else
(M4A/AAC, MP4, WebM/Opus, ...). One resampler (soxr_hq) everywhere, identical to training (ml/audio_cache.py).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

SR = 16000


class DecodeError(ValueError):
    pass


def ffmpeg_exe() -> str | None:
    env = os.getenv("FFMPEG_PATH")
    if env and (Path(env).is_file() or shutil.which(env)):
        return shutil.which(env) or env
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return shutil.which("ffmpeg")


def resample(x: np.ndarray, sr: int, target: int = SR) -> np.ndarray:
    if sr == target:
        return x.astype(np.float32)
    import soxr

    return soxr.resample(x.astype(np.float32), sr, target, quality="HQ").astype(np.float32)


def to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x


def _ffmpeg_decode(path: str) -> np.ndarray:
    exe = ffmpeg_exe()
    if not exe:
        raise DecodeError("ffmpeg not available to decode this format")
    cmd = [exe, "-v", "error", "-nostdin", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    if r.returncode != 0 or not r.stdout:
        raise DecodeError("could not decode audio: " + r.stderr.decode(errors="ignore")[-300:])
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def load_audio(path: str | Path) -> tuple[np.ndarray, int, str]:
    """Returns (float32 mono 16 kHz, original sample rate or -1 if unknown, decoder name)."""
    path = str(path)
    try:
        import soundfile as sf

        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = resample(to_mono(x), sr)
        dec = "soundfile"
    except Exception:  # noqa: BLE001
        x, sr, dec = _ffmpeg_decode(path), -1, "ffmpeg"
    if x.size == 0:
        raise DecodeError("empty audio")
    if not np.isfinite(x).all():
        x = np.nan_to_num(x)
    return np.clip(x, -1.0, 1.0).astype(np.float32), sr, dec


def pcm_bytes_to_float(buf: bytes, encoding: str) -> np.ndarray:
    if encoding == "s16le":
        return np.frombuffer(buf[: len(buf) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    return np.frombuffer(buf[: len(buf) // 4 * 4], dtype="<f4").astype(np.float32)
