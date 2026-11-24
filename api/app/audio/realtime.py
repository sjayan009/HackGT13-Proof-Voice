"""Cheap per-window signal quality for live streams (energy VAD, SNR, level, clipping). Pure numpy, ~0.1 ms/window.

Used identically by the live call path (api/app/telephony), the Aegis scheduler gate and offline call-level
evaluation (ml/telephony_eval) so that "usable speech" means the same thing everywhere.
"""
from __future__ import annotations

import numpy as np

SR = 16000
FRAME = 320  # 20 ms
SILENCE_DBFS = -50.0
VAD_MARGIN_DB = 10.0
DIGITAL_ZERO_DB = -100.0  # zero-filled lost packets / digital silence: not a measurement of the noise floor


def frame_db(x: np.ndarray) -> np.ndarray:
    n = len(x) // FRAME
    if n == 0:
        return np.zeros(0, np.float32)
    f = x[: n * FRAME].reshape(n, FRAME).astype(np.float32)
    return 20 * np.log10(np.sqrt(np.mean(f * f, axis=1)) + 1e-9)


def window_quality(x: np.ndarray, floor_db: float | None = None) -> dict:
    """speech_s: frames above max(noise floor + 10 dB, -50 dBFS). floor_db: call-level noise floor if known
    (a 3 s window full of speech overestimates its own floor); else the window's 10th-percentile frame level."""
    db = frame_db(x)
    if db.size == 0:
        return {"speech_s": 0.0, "snr_db": 0.0, "level_dbfs": -120.0, "clip_ratio": 0.0, "floor_db": -120.0}
    real = db[db > DIGITAL_ZERO_DB]
    if floor_db is None and real.size == 0:
        return {"speech_s": 0.0, "snr_db": 0.0, "level_dbfs": -120.0, "clip_ratio": 0.0, "floor_db": -120.0}
    floor = float(np.percentile(real, 10)) if floor_db is None else float(floor_db)
    speech = db > max(floor + VAD_MARGIN_DB, SILENCE_DBFS)
    snr = float(np.median(db[speech]) - floor) if speech.any() else 0.0
    rms = float(np.sqrt(np.mean(np.asarray(x, np.float32) ** 2)))
    return {"speech_s": round(float(speech.sum()) * FRAME / SR, 3), "snr_db": round(snr, 1),
            "level_dbfs": round(20 * np.log10(rms + 1e-9), 1),
            "clip_ratio": round(float(np.mean(np.abs(x) > 0.99)), 4), "floor_db": round(floor, 1)}


class FloorTracker:
    """Call-level noise floor: slow-rising, fast-falling minimum of the 10th-percentile frame level per hop."""

    def __init__(self):
        self.floor: float | None = None

    def update(self, hop: np.ndarray) -> float | None:
        db = frame_db(hop)
        db = db[db > DIGITAL_ZERO_DB]  # lost packets are zero-filled upstream; they must not drag the floor down
        if db.size == 0:
            return self.floor
        p10 = float(np.percentile(db, 10))
        self.floor = p10 if self.floor is None else (p10 if p10 < self.floor else self.floor + 0.05 * (p10 - self.floor))
        return self.floor


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = np.arange(3 * SR) / SR
    tone = 0.3 * np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 2 * t) > 0)  # 50% duty "speech"
    q = window_quality(tone + 0.003 * rng.standard_normal(len(t)))
    assert 1.2 < q["speech_s"] < 1.8 and q["snr_db"] > 25, q
    assert window_quality(np.zeros(3 * SR))["speech_s"] == 0.0
    noisy = tone + 0.003 * rng.standard_normal(len(t))
    lossy = noisy.copy()
    for k in range(0, len(lossy), 3200):  # 10 % of 20 ms frames zero-filled (lost packets)
        lossy[k:k + 320] = 0
    ft = FloorTracker()
    for k in range(0, len(lossy), 8000):
        ft.update(lossy[k:k + 8000])
    q2 = window_quality(lossy, ft.floor)
    assert abs(q2["snr_db"] - q["snr_db"]) < 5 and ft.floor > -70, (q2, ft.floor)
    print("realtime quality ok", q)
