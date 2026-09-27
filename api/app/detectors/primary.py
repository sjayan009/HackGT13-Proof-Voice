"""Primary anti-spoofing detector: the frozen, selected SSL model + its calibration.

Model directory layout (written by ml/train.py + ml/calibrate.py):
    model.pt, meta.json, backbone_config/, calibration.json
calibration.json: {"a": float, "b": float, "decision_threshold": float, ...}
    p_display = sigmoid(a * logit + b)  (Platt scaling fitted on validation; monotone -> minDCF unchanged)
    decision_threshold: p_display threshold minimising validation minDCF.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from app.detectors.ssl_model import load_detector, normalize

SR = 16000


@dataclass
class DetectorResult:
    synthetic_probability: float
    raw_score: float  # logit
    latency_ms: float
    diagnostics: dict = field(default_factory=dict)


class PrimaryDetector:
    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.model, self.meta = load_detector(self.model_dir, self.device)
        if self.device.type == "cuda":
            self.model.half()
        cal_path = self.model_dir / "calibration.json"
        self.cal = json.loads(cal_path.read_text()) if cal_path.exists() else {"a": 1.0, "b": 0.0,
                                                                                "decision_threshold": 0.5}
        self.name = self.meta.get("name", self.meta.get("backbone", "ssl-detector"))
        self.version = str(self.meta.get("version", self.meta.get("step", "0")))
        self._lock = threading.Lock()  # one GPU forward at a time

    @property
    def threshold(self) -> float:
        return float(self.cal["decision_threshold"])

    def calibrate(self, logit: np.ndarray | float) -> np.ndarray | float:
        return 1.0 / (1.0 + np.exp(-(self.cal["a"] * np.asarray(logit) + self.cal["b"])))

    def logits(self, clips: list[np.ndarray], bs: int = 16) -> np.ndarray:
        try:
            return self._logits(clips, bs)
        except RuntimeError as e:  # cuDNN workspace failures seen on 8 GB WDDM GPUs -> retry without cuDNN
            if self.device.type != "cuda":
                raise
            torch.cuda.empty_cache()
            torch.backends.cudnn.enabled = False
            return self._logits(clips, max(1, bs // 2))

    @torch.no_grad()
    def _logits(self, clips: list[np.ndarray], bs: int = 16) -> np.ndarray:
        out = np.zeros(len(clips), np.float32)
        order = np.argsort([len(c) for c in clips])
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        with self._lock:
            for b in range(0, len(order), bs):
                ids = order[b:b + bs]
                lens = torch.tensor([max(len(clips[i]), 400) for i in ids], device=self.device)
                xb = torch.zeros(len(ids), int(lens.max()), device=self.device)
                for j, i in enumerate(ids):
                    c = clips[i] if len(clips[i]) >= 400 else np.pad(clips[i], (0, 400 - len(clips[i])))
                    xb[j, :len(c)] = torch.from_numpy(np.ascontiguousarray(c))
                xb = normalize(xb, lens).to(dtype)
                out[ids] = self.model(xb, lens).float().cpu().numpy()
        return out

    def clip_logits(self, clips: list[np.ndarray], bs: int = 16) -> np.ndarray:
        """Clip-level logit: clips <= 4 s scored whole; longer clips = mean logit of 4 s windows (hop 2 s).
        Chosen on full-length validation clips (outputs/results/aggregation.json): mean 0.016 vs full 0.021 minDCF."""
        w, h = 4 * SR, 2 * SR
        segs, owner = [], []
        for k, x in enumerate(clips):
            if len(x) <= w:
                starts = [0]
            else:
                starts = list(range(0, len(x) - w + 1, h))
                if starts[-1] + w < len(x):
                    starts.append(len(x) - w)
            for s in starts:
                segs.append(x[s:s + w])
                owner.append(k)
        lg = self.logits(segs, bs)
        owner = np.array(owner)
        return np.array([lg[owner == k].mean() for k in range(len(clips))], np.float32)

    def score(self, audio: np.ndarray, sr: int = SR) -> DetectorResult:
        assert sr == SR, "resample to 16 kHz first"
        t = time.perf_counter()
        lg = float(self.clip_logits([audio])[0])
        return DetectorResult(float(self.calibrate(lg)), lg, (time.perf_counter() - t) * 1000)

    def score_windows(self, audio: np.ndarray, win_ms: int = 2000, hop_ms: int = 500) -> list[dict]:
        w, h = int(SR * win_ms / 1000), int(SR * hop_ms / 1000)
        if len(audio) <= w:
            starts = [0]
        else:
            starts = list(range(0, len(audio) - w + 1, h))
            if starts[-1] + w < len(audio):
                starts.append(len(audio) - w)
        clips = [audio[s:s + w] for s in starts]
        lg = self.logits(clips)
        p = self.calibrate(lg)
        return [{"start_ms": int(s * 1000 / SR), "end_ms": int(min(len(audio), s + w) * 1000 / SR),
                 "synthetic_probability": float(pi), "logit": float(li)} for s, pi, li in zip(starts, p, lg)]
