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


class TensorRTBackend:
    """TensorRT FP16 engine (engine/trt_build.py) for fixed 3.0 s windows, dynamic batch 1-16.

    Engine is exported/built on first use and cached under <model_dir>/trt/. Normalisation is folded into the
    engine; the lengths/attention-mask input is dropped, so every clip MUST be exactly 48000 samples (Aegis only
    sends full 3 s windows from telephony/media.py). Pinned host buffers + one CUDA stream, sized for max batch.
    Not thread-safe: Aegis calls it from its single GPU worker thread.
    """

    def __init__(self, model_dir):
        import tensorrt as trt
        import torch

        from engine.trt_build import MAX_B, WIN, ensure_engine

        self.win, self.max_b = WIN, MAX_B
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(
            ensure_engine(model_dir).read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT engine failed to deserialize (TRT/driver mismatch? delete models/*/trt/)")
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.h_in = torch.empty(MAX_B, WIN, dtype=torch.float32, pin_memory=True)
        self.h_out = torch.empty(MAX_B, dtype=torch.float32, pin_memory=True)
        self.d_in = torch.empty(MAX_B, WIN, dtype=torch.float32, device="cuda")
        self.d_out = torch.empty(MAX_B, dtype=torch.float32, device="cuda")
        self.ctx.set_tensor_address("wav", self.d_in.data_ptr())
        self.ctx.set_tensor_address("logit", self.d_out.data_ptr())
        self.name = "tensorrt-fp16"
        self.device_name = torch.cuda.get_device_name(0)

    def logits(self, clips: list[np.ndarray]) -> np.ndarray:
        import torch

        assert all(len(c) == self.win for c in clips), "TensorRTBackend scores full 3.0 s (48000-sample) windows only"
        out = np.empty(len(clips), np.float32)
        with torch.cuda.stream(self.stream):
            for s in range(0, len(clips), self.max_b):
                part = clips[s:s + self.max_b]
                n = len(part)
                self.h_in[:n].numpy()[:] = np.stack(part)
                self.d_in[:n].copy_(self.h_in[:n], non_blocking=True)
                self.ctx.set_input_shape("wav", (n, self.win))
                if not self.ctx.execute_async_v3(self.stream.cuda_stream):
                    raise RuntimeError("TensorRT execute failed")
                self.h_out[:n].copy_(self.d_out[:n], non_blocking=True)
                self.stream.synchronize()
                out[s:s + n] = self.h_out[:n].numpy()
        return out
