"""Streaming analysis: ring buffer at 16 kHz, hop-driven scoring, rolling decision, time-to-confidence.

Per hop (default 0.5 s of new audio, once >= 2.0 s buffered):
  * synthetic_probability: detector on the most recent window (default 3 s; local evidence -> timeline)
  * rolling_probability:   detector on the most recent <= 8 s (accumulated evidence, same regime as file scoring)
  * status / analysis_confidence from the rolling score, speech duration and window agreement
  * time_to_confidence_ms: stream time at which the current decisive status began, if it has held since
"""
from __future__ import annotations

import numpy as np

from app.orchestration.pipeline import status_for

SR = 16000
ROLL_S = 8.0
MAX_BUFFER_S = 120.0
SILENCE_DBFS = -50.0


class StreamAnalyzer:
    def __init__(self, detector, in_sr: int, win_ms: int = 2000, hop_ms: int = 500):
        import soxr

        self.det = detector
        self.in_sr = int(in_sr)
        self.rs = soxr.ResampleStream(self.in_sr, SR, 1, dtype="float32", quality="HQ") if self.in_sr != SR else None
        self.buf = np.zeros(0, np.float32)
        self.total = 0  # samples received (16 kHz) over the whole stream; buffer itself is capped
        self.win, self.hop = int(SR * win_ms / 1000), int(SR * hop_ms / 1000)
        self.next_at = int(2.0 * SR)  # first estimate at 2 s; confidence reflects < 3 s of evidence
        self.speech_samples = 0
        self.window_ps: list[float] = []
        self.ttc_start: int | None = None
        self.decisive: str | None = None

    def push(self, x: np.ndarray) -> None:
        x = np.asarray(x, np.float32)
        if self.rs is not None:
            x = self.rs.resample_chunk(x)
        self.total += len(x)
        self.buf = np.concatenate([self.buf, x])[-int(MAX_BUFFER_S * SR):]

    def flush(self) -> None:
        """Drain the streaming resampler's internal delay at end of stream."""
        if self.rs is not None:
            tail = self.rs.resample_chunk(np.zeros(0, np.float32), last=True)
            self.total += len(tail)
            self.buf = np.concatenate([self.buf, tail])[-int(MAX_BUFFER_S * SR):]

    def ready(self) -> bool:
        return self.total >= self.next_at

    def step(self) -> dict:
        """Score the newest hop. Blocking (run in a thread)."""
        n = len(self.buf)
        self.next_at = self.total + self.hop
        seg = self.buf[max(0, n - self.win):]
        hop_seg = self.buf[max(0, n - self.hop):]
        lvl = float(20 * np.log10(np.sqrt(np.mean(hop_seg ** 2)) + 1e-9))
        if lvl > SILENCE_DBFS:
            self.speech_samples += len(hop_seg)
        speech_ratio = min(1.0, self.speech_samples / max(1, self.total))
        roll = self.buf[max(0, n - int(ROLL_S * SR)):]
        lg = self.det.logits([seg, roll])
        p_win, p_roll = (float(v) for v in self.det.calibrate(lg))
        self.window_ps.append(p_win)
        thr = self.det.threshold
        speech_s = self.speech_samples / SR
        recent = np.array(self.window_ps[-6:])
        agree = float(np.mean((recent >= thr) == (p_roll >= thr)))
        margin = float(np.clip(abs(p_roll - thr) / max(thr, 1 - thr) * 1.5, 0, 1))
        conf = round((0.35 * min(1.0, speech_s / 3.0) + 0.4 * margin + 0.25 * agree) * min(1.0, self.total / (3.0 * SR)), 3)
        status = status_for(p_roll, thr, speech_s, conf) if lvl > SILENCE_DBFS or speech_s >= 1 else "insufficient_evidence"
        t_ms = int(self.total * 1000 / SR)
        if status in ("likely_human", "likely_synthetic") and conf >= 0.5:
            if self.decisive != status:
                self.decisive, self.ttc_start = status, t_ms
        else:
            self.decisive, self.ttc_start = None, None
        return {"type": "analysis.window", "t_ms": t_ms, "start_ms": int(max(0, self.total - self.win) * 1000 / SR),
                "end_ms": t_ms, "synthetic_probability": round(p_win, 4), "rolling_probability": round(p_roll, 4),
                "analysis_confidence": conf, "status": status, "time_to_confidence_ms": self.ttc_start,
                "level_dbfs": round(lvl, 1), "speech_ratio": round(speech_ratio, 3)}

    def audio(self) -> np.ndarray:
        return self.buf.copy()
