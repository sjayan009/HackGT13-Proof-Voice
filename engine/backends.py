"""Inference backends for Aegis. Interface: `.name`, `.logits(list[np.ndarray]) -> sequence of float logits`."""
from __future__ import annotations

import numpy as np


class TorchBackend:
    """The serving PrimaryDetector (PyTorch; FP16 on CUDA, FP32 on CPU). One forward per microbatch."""

    def __init__(self, detector):
        self.det = detector
        cuda = detector.device.type == "cuda"
        self.name = "pytorch-fp16" if cuda else "pytorch-fp32"
        self.device_name = "cpu"
        if cuda:
            import torch

            self.device_name = torch.cuda.get_device_name(detector.device)

    def logits(self, clips: list[np.ndarray]) -> np.ndarray:
        return self.det.logits(clips, bs=max(1, len(clips)))
