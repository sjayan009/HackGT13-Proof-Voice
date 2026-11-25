"""Backend latency + prediction parity for the Aegis inference backends (fixed 3.0 s windows).

    .venv/Scripts/python.exe benchmarks/latency/backend_latency.py [--iters 100] [--n-parity 256]

Backends: pytorch-fp32, pytorch-fp16 (serving PrimaryDetector), tensorrt-fp16 (engine/backends.TensorRTBackend).
All three backends stay resident and are interleaved per batch size (fair under GPU co-tenants). Timing is the full backend call as Aegis
sees it (numpy in -> H2D -> forward -> D2H -> numpy out, CUDA-synchronised). Writes outputs/benchmarks/backend_latency.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "api")]

from app.audio.ingest import load_audio  # noqa: E402
from app.detectors.primary import PrimaryDetector  # noqa: E402
from app.detectors.ssl_model import load_detector, normalize  # noqa: E402

MODEL = ROOT / "models" / "selected"
WIN = 48000
BATCHES = (1, 2, 4, 8, 16)


class TorchFP32:
    """FP32 reference on GPU (PrimaryDetector always halves on CUDA), same normalise -> forward path."""

    def __init__(self):
        self.model, _ = load_detector(MODEL, "cuda")
        self.name = "pytorch-fp32"

    @torch.no_grad()
    def logits(self, clips):
        out = []
        for s in range(0, len(clips), 16):
            x = torch.from_numpy(np.stack(clips[s:s + 16])).cuda()
            lens = torch.full((len(x),), WIN, device="cuda")
            out.append(self.model(normalize(x, lens), lens).float().cpu().numpy())
        return np.concatenate(out)


class TorchFP16:
    def __init__(self):
        self.det = PrimaryDetector(MODEL, "cuda")
        self.name = "pytorch-fp16"

    def logits(self, clips):
        return self.det.logits(clips, bs=16)


def make_trt():
    from engine.backends import TensorRTBackend

    return TensorRTBackend(MODEL)


def gpu_snapshot(samples: int = 5) -> dict:
    """Other GPU processes (WDDM hides per-process memory) + utilisation/memory sampled over ~samples*0.2 s."""
    q = lambda *a: subprocess.run(["nvidia-smi", *a, "--format=csv,noheader,nounits"], capture_output=True,
                                  text=True).stdout.strip()
    procs = []
    for line in filter(None, q("--query-compute-apps=pid,process_name").splitlines()):
        pid, name = [s.strip() for s in line.split(",", 1)]
        if int(pid) == os.getpid():
            continue
        cmd = subprocess.run(["powershell", "-NoProfile", "-c",
                              f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
                             capture_output=True, text=True).stdout.strip() if platform.system() == "Windows" else ""
        procs.append({"pid": int(pid), "process": Path(name).name, "cmd": cmd.split("python.exe\"")[-1].strip()[:160]})
    util, mem = [], []
    for _ in range(samples):
        u, m = q("--query-gpu=utilization.gpu,memory.used").split(",")
        util.append(int(u)), mem.append(int(m))
        time.sleep(0.2)
    return {"processes": procs, "util_pct_mean": round(float(np.mean(util)), 1), "util_pct_max": max(util),
            "mem_used_mib": max(mem)}


def nvml_used_mib() -> float:
    import pynvml

    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).used / 2**20


def load_windows(n: int) -> tuple[list[np.ndarray], np.ndarray]:
    """n real 3.0 s windows (centre crop) from validation, class-balanced, fixed seed."""
    df = pd.read_csv(ROOT / "outputs" / "manifests" / "val.csv")
    df = df[df.duration >= 3.05]
    df = pd.concat([g.sample(min(len(g), n // 2 + 16), random_state=0) for _, g in df.groupby("y")])
    clips, ys = [], []
    for y in (0, 1):
        k = 0
        for p in df[df.y == y].path:
            try:
                x, _, _ = load_audio(ROOT / p)
            except Exception:  # noqa: BLE001
                continue
            if len(x) < WIN:
                continue
            s = (len(x) - WIN) // 2
            clips.append(x[s:s + WIN].copy()), ys.append(y)
            k += 1
            if k == n // 2:
                break
    return clips, np.array(ys)


def timed(be, clips, n: int) -> list[float]:
    t = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        be.logits(clips)
        torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) * 1000)
    return t


def bench(bes: list, pool: list[np.ndarray], iters: int, warmup: int, rounds: int = 4) -> tuple[list[dict], list]:
    """All backends resident; per batch size the backends alternate in `rounds` slices so a bursty co-tenant on
    the GPU hits each backend equally. Torch peak VRAM = peak during its own slices minus the other backends'
    resident torch memory; TRT VRAM = NVML device-memory delta from its load (TRT allocates outside torch)."""
    rows, snaps = [], []
    for b in BATCHES:
        clips = pool[:b]
        snaps.append(gpu_snapshot(3) | {"at": f"batch {b}"})
        t = {be.name: [] for be in bes}
        peak = {be.name: 0 for be in bes}
        for be in bes:
            timed(be, clips, warmup)
        for _ in range(rounds):
            for be in bes:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t[be.name] += timed(be, clips, iters // rounds)
                peak[be.name] = max(peak[be.name], torch.cuda.max_memory_allocated())
        for be in bes:
            x = np.array(t[be.name])
            p50 = float(np.percentile(x, 50))
            row = {"backend": be.name, "batch": b, "iters": len(x), "p50_ms": round(p50, 2),
                   "p95_ms": round(float(np.percentile(x, 95)), 2), "p99_ms": round(float(np.percentile(x, 99)), 2),
                   "mean_ms": round(float(x.mean()), 2), "ms_per_window": round(p50 / b, 2),
                   "windows_per_s": round(b * 1000 / p50, 1)}
            if be.name.startswith("pytorch"):
                others = sum(o.torch_resident for o in bes if o is not be)
                row |= {"peak_vram_mib": round((peak[be.name] - others) / 2**20), "vram_source": "torch"}
            else:
                row |= {"peak_vram_mib": round(be.nvml_mib), "vram_source": "nvml_delta"}
            rows.append(row)
            print(row, flush=True)
    return rows, snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--n-parity", type=int, default=256)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "benchmarks" / "backend_latency.json"))
    a = ap.parse_args()

    import pynvml
    import tensorrt as trt

    pynvml.nvmlInit()
    before = gpu_snapshot()
    print("gpu before:", before, flush=True)
    clips, ys = load_windows(a.n_parity)
    print(f"parity windows: {len(clips)} ({int(ys.sum())} spoof)", flush=True)

    logits, bes = {}, []
    torch.cuda.init()
    for make in (TorchFP32, TorchFP16, make_trt):
        torch.cuda.synchronize()
        base_nvml, base_torch = nvml_used_mib(), torch.cuda.memory_allocated()
        be = make()
        logits[be.name] = np.asarray(be.logits(clips), np.float32)
        torch.cuda.synchronize(), torch.cuda.empty_cache()
        be.torch_resident = torch.cuda.memory_allocated() - base_torch
        be.nvml_mib = nvml_used_mib() - base_nvml
        bes.append(be)
    results, during = bench(bes, clips, a.iters, a.warmup)

    cal = json.loads((MODEL / "calibration.json").read_text())
    dec = lambda lg: 1 / (1 + np.exp(-(cal["a"] * lg + cal["b"]))) >= cal["decision_threshold"]
    parity = []
    for cand, ref in (("tensorrt-fp16", "pytorch-fp16"), ("tensorrt-fp16", "pytorch-fp32"),
                      ("pytorch-fp16", "pytorch-fp32")):
        d = np.abs(logits[cand] - logits[ref])
        parity.append({"candidate": cand, "reference": ref, "n_windows": len(clips),
                       "max_abs_diff": round(float(d.max()), 4), "mean_abs_diff": round(float(d.mean()), 4),
                       "p99_abs_diff": round(float(np.percentile(d, 99)), 4),
                       "decision_agreement": round(float((dec(logits[cand]) == dec(logits[ref])).mean()), 4),
                       "decision_flips": int((dec(logits[cand]) != dec(logits[ref])).sum())})
    for r in parity:
        print(r)

    busy = before["util_pct_max"] > 10 or any(s["processes"] for s in [before, *during])
    others = sorted({f'{p["process"]} {p["cmd"]}'.strip() for s in [before, *during] for p in s["processes"]})
    contention = ("GPU shared during the run with: " + "; ".join(others) +
                  f". Idle-check utilisation before measuring: mean {before['util_pct_mean']}%, max "
                  f"{before['util_pct_max']}%. Absolute latencies are inflated vs an idle GPU; backends were interleaved in "
                  "slices per batch size so each saw the same co-tenant load, so ratios are the meaningful number.") if busy else \
        "none observed (no other compute processes; GPU idle before measuring)"

    out = {
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "machine": {"gpu": torch.cuda.get_device_name(0), "driver": pynvml.nvmlSystemGetDriverVersion(),
                    "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "tensorrt": trt.__version__,
                    "python": platform.python_version(), "os": platform.platform()},
        "setup": {"window_samples": WIN, "sample_rate": 16000, "warmup": a.warmup, "iters": a.iters,
                  "timing": "end-to-end backend.logits(list[np.ndarray]) incl. H2D/D2H, CUDA-synchronised",
                  "model": "models/selected (XLS-R 300M, 12 layers)"},
        "contention": contention,
        "gpu_snapshots": [before | {"at": "before run"}] + during,
        "results": results,
        "parity": {"threshold": cal["decision_threshold"], "windows": "class-balanced centre-cropped 3.0 s "
                   "windows from outputs/manifests/val.csv (seed 0)", "rows": parity},
    }
    for s in out["gpu_snapshots"]:  # flatten for the generic table renderer
        s["processes"] = "; ".join(f'{p["pid"]} {p["process"]} {p["cmd"]}' for p in s["processes"]) or "none"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
