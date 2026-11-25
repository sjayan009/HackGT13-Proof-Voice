"""ONNX export + TensorRT FP16 engine build for the serving detector (fixed 3.0 s windows, dynamic batch 1-16).

    python -m engine.trt_build [--model models/selected]      # writes <model>/trt/{model_fp16.onnx, *.engine}

Graph contract: input `wav` float32 [B, 48000] RAW audio (normalisation is folded into the graph, in fp32,
exactly as `normalize()` for full-length rows), output `logit` float32 [B]. All rows are assumed full length,
so the lengths / attention-mask input is dropped (an all-ones mask is numerically a no-op). Aegis only ever
sends full 3 s windows; TensorRTBackend asserts it.

TensorRT >= 10.12 deprecates (11 removes) the FP16 builder flag: networks are strongly typed, so FP16 lives in
the ONNX graph itself (model weights/activations exported in half, normalisation + softmax-pool in fp32).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIN = 48000
MIN_B, OPT_B, MAX_B = 1, 8, 16


def paths(model_dir: str | Path) -> tuple[Path, Path]:
    import tensorrt as trt

    d = Path(model_dir) / "trt"
    return d / "model_fp16.onnx", d / f"model_fp16_b{MIN_B}-{MAX_B}_trt{trt.__version__}.engine"


def export_onnx(model_dir: str | Path, onnx_path: Path) -> None:
    import torch

    sys.path.insert(0, str(ROOT / "api"))
    from app.detectors.ssl_model import load_detector

    model, _ = load_detector(model_dir, "cuda")
    model.half()

    class Wrapped(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, wav):  # float32 [B, WIN], full-length rows
            mu = wav.mean(1, keepdim=True)
            sd = ((wav - mu) ** 2).mean(1, keepdim=True).sqrt()
            x = ((wav - mu) / (sd + 1e-7)).half()
            # SSLDetector.forward with an all-ones mask, minus the mask ops (no-ops for full rows, and the HF
            # mask path emits fp32 casts that a strongly typed TRT network rejects)
            m = self.m
            hs = torch.stack(m.ssl(x, attention_mask=None, output_hidden_states=True).hidden_states, 0)
            h = m.proj((torch.softmax(m.layer_w, 0).to(hs.dtype)[:, None, None, None] * hs).sum(0))
            mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
            return m.head(m.pool(h, mask)).squeeze(-1).float()

    # The stock opset-17 layer_norm symbolic upcasts fp16 input but not scale/bias (TRT rejects the mix) and the
    # fp32 output then promotes the whole encoder to fp32. Mirror PyTorch instead: fp32 LayerNorm, fp16 out.
    from torch.onnx import register_custom_op_symbolic, symbolic_helper

    @symbolic_helper.parse_args("v", "is", "v", "v", "f", "b")
    def layer_norm(g, x, shape, w, b, eps, _cudnn):
        f32 = [g.op("Cast", t, to_i=1) for t in (x, w, b)]
        return g.op("Cast", g.op("LayerNormalization", *f32, axis_i=-len(shape), epsilon_f=eps), to_i=10)

    register_custom_op_symbolic("aten::layer_norm", layer_norm, 17)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    x = torch.randn(2, WIN, device="cuda")
    with torch.no_grad():
        torch.onnx.export(Wrapped(model).eval(), (x,), str(onnx_path), input_names=["wav"], output_names=["logit"],
                          dynamic_axes={"wav": {0: "B"}, "logit": {0: "B"}}, opset_version=17, dynamo=False)


def build_engine(onnx_path: Path, engine_path: Path) -> None:
    import tensorrt as trt

    log = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(log)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(net, log)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    prof = builder.create_optimization_profile()
    prof.set_shape("wav", (MIN_B, WIN), (OPT_B, WIN), (MAX_B, WIN))
    cfg.add_optimization_profile(prof)
    plan = builder.build_serialized_network(net, cfg)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_path.write_bytes(bytes(plan))


def ensure_engine(model_dir: str | Path) -> Path:
    """Export + build on first use; cached under <model_dir>/trt/ (keyed by TRT version; delete to rebuild)."""
    onnx_path, engine_path = paths(model_dir)
    if not engine_path.exists():
        if not onnx_path.exists():
            t = time.time()
            export_onnx(model_dir, onnx_path)
            print(f"[trt] exported {onnx_path} in {time.time() - t:.0f}s", flush=True)
        t = time.time()
        build_engine(onnx_path, engine_path)
        print(f"[trt] built {engine_path} in {time.time() - t:.0f}s", flush=True)
    return engine_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "selected"))
    print(ensure_engine(ap.parse_args().model))
