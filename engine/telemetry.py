"""Aegis telemetry: rolling latency/batch/action stats, NVML GPU sampling, Prometheus text exposition.

Every number here is measured in-process; GPU fields are None when NVML is unavailable (never fabricated).
"""
from __future__ import annotations

import threading
import time
from collections import Counter, deque

import numpy as np

ACTIONS = ("infer", "escalate", "skip_silence", "skip_clear", "shed")


def _pct(xs, qs=(50, 95, 99)) -> dict:
    if not xs:
        return {f"p{q}": None for q in qs} | {"n": 0}
    a = np.fromiter(xs, float)
    return {f"p{q}": round(float(np.percentile(a, q)), 2) for q in qs} | {"n": int(a.size)}


class Gpu:
    """NVML sampler (1 Hz background thread)."""

    def __init__(self, index: int = 0, period_s: float = 1.0):
        self.latest: dict | None = None
        self.name: str | None = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nv, self._h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(index)
            self.name = pynvml.nvmlDeviceGetName(self._h)
        except Exception:  # noqa: BLE001 - no NVIDIA driver / no pynvml -> gpu stays None
            self._nv = None
            return
        self.period = period_s
        threading.Thread(target=self._loop, daemon=True, name="aegis-nvml").start()

    def sample(self) -> dict | None:
        if self._nv is None:
            return None
        nv, h = self._nv, self._h
        mem = nv.nvmlDeviceGetMemoryInfo(h)
        out = {"util_pct": nv.nvmlDeviceGetUtilizationRates(h).gpu, "mem_used_mb": mem.used // 2**20,
               "mem_total_mb": mem.total // 2**20, "power_w": None, "temp_c": None}
        try:
            out["power_w"] = round(nv.nvmlDeviceGetPowerUsage(h) / 1000, 1)
            out["temp_c"] = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
        except Exception:  # noqa: BLE001 - some laptop GPUs do not expose power
            pass
        return out

    def _loop(self):
        while True:
            try:
                self.latest = self.sample()
            except Exception:  # noqa: BLE001
                self.latest = None
            time.sleep(self.period)


class Telemetry:
    def __init__(self, window: int = 2000, gpu: Gpu | None = None):
        self.lat = deque(maxlen=window)      # submit -> result, ms
        self.compute = deque(maxlen=window)  # batch forward time, ms (per batch)
        self.wait = deque(maxlen=window)     # queue wait, ms (per request)
        self.batches = deque(maxlen=window)  # batch sizes
        self.hops = deque(maxlen=20000)      # (t, action) for rate computation
        self.actions = Counter()
        self.gpu = gpu
        self._lock = threading.Lock()

    def hop(self, action: str) -> None:
        with self._lock:
            self.actions[action] += 1
            self.hops.append((time.monotonic(), action))

    def request_done(self, latency_ms: float, wait_ms: float) -> None:
        with self._lock:
            self.lat.append(latency_ms)
            self.wait.append(wait_ms)

    def batch_done(self, size: int, compute_ms: float) -> None:
        with self._lock:
            self.batches.append(size)
            self.compute.append(compute_ms)

    def reset(self) -> None:
        with self._lock:
            for d in (self.lat, self.compute, self.wait, self.batches, self.hops):
                d.clear()
            self.actions.clear()

    def snapshot(self, **extra) -> dict:
        with self._lock:
            now = time.monotonic()
            recent = [a for t, a in self.hops if now - t <= 10.0]
            span = min(10.0, max(1e-6, now - self.hops[0][0])) if self.hops else 10.0
            bs = list(self.batches)
            hist = Counter(bs)
            snap = {
                "latency_ms": _pct(self.lat), "compute_ms": _pct(self.compute, (50, 95)),
                "queue_wait_ms": _pct(self.wait, (50, 95)),
                "windows_per_s": {"demand": round(len(recent) / span, 2),
                                  "inferred": round(sum(a in ("infer", "escalate") for a in recent) / span, 2)},
                "actions": {a: self.actions.get(a, 0) for a in ACTIONS},
                "batch_size": {"mean": round(float(np.mean(bs)), 2) if bs else None, "max": max(bs) if bs else None,
                               "hist": {str(k): v for k, v in sorted(hist.items())}},
            }
        snap["gpu"] = self.gpu.latest if self.gpu else None
        snap["t"] = time.time()
        return extra | snap

    @staticmethod
    def prometheus(s: dict) -> str:
        """Prometheus text format for the snapshot (hand-written; no client library needed)."""
        lines = []

        def g(name, val, help_, labels=""):
            if val is None:
                return
            lines.append(f"# HELP proofvoice_{name} {help_}\n# TYPE proofvoice_{name} gauge")
            lines.append(f"proofvoice_{name}{labels} {val}")

        g("active_calls", s.get("active_calls"), "Calls currently streaming")
        g("queue_depth", s.get("queue_depth"), "Inference requests waiting")
        for q in ("p50", "p95", "p99"):
            g(f"latency_ms_{q}", s["latency_ms"][q], f"Per-window submit-to-result latency {q} (ms)")
        g("compute_ms_p50", s["compute_ms"]["p50"], "Per-batch forward time p50 (ms)")
        g("windows_demand_per_s", s["windows_per_s"]["demand"], "Hops arriving per second")
        g("windows_inferred_per_s", s["windows_per_s"]["inferred"], "Windows run on the model per second")
        g("batch_size_mean", s["batch_size"]["mean"], "Mean microbatch size")
        lines.append("# HELP proofvoice_hops_total Hops by scheduler action\n# TYPE proofvoice_hops_total counter")
        lines += [f'proofvoice_hops_total{{action="{a}"}} {n}' for a, n in s["actions"].items()]
        gpu = s.get("gpu") or {}
        g("gpu_util_pct", gpu.get("util_pct"), "GPU utilisation (NVML)")
        g("gpu_mem_used_mb", gpu.get("mem_used_mb"), "GPU memory used (NVML, MiB)")
        g("gpu_power_w", gpu.get("power_w"), "GPU power draw (NVML, W)")
        return "\n".join(lines) + "\n"
