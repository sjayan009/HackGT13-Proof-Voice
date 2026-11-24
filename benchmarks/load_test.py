"""Concurrent-call load test for Aegis on the real detector (PRD §8, §11).

N simulated callers stream real validation audio through the production Twilio path (TwilioStream -> CallSession
-> Aegis -> risk engine) at real-time pace, for each scheduling policy. Measures per-window latency, GPU utilisation
and memory (NVML), compute actually spent, shedding, and call-level outcomes under load.

    .venv/Scripts/python.exe benchmarks/load_test.py --calls 1 10 20 30 40 --seconds 20

Writes outputs/benchmarks/load_test.json. Numbers are from this machine at this time; other GPU work running
concurrently is recorded in the output ("gpu_baseline") and invalidates absolute latency comparisons.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "api"), str(ROOT)]

from app import config  # noqa: E402
from app.runtime import RT  # noqa: E402


def setup(policy: str):
    from engine.aegis import Aegis
    from engine.backends import TorchBackend
    from engine.telemetry import Gpu, Telemetry

    from app.detectors.primary import PrimaryDetector
    from app.risk.engine import Calibration, RiskPolicy
    from app.verification.alerts import Hub
    from app.verification.store import Store

    if RT.detector is None:
        RT.detector = PrimaryDetector(config.MODEL_DIR, config.resolve_device())
        RT.detector.score(np.zeros(16000, np.float32))
        RT.telemetry = Telemetry(gpu=Gpu(period_s=0.25))
        RT.aegis = Aegis(TorchBackend(RT.detector), RT.telemetry, policy, config.AEGIS_MAX_BATCH,
                         config.AEGIS_MAX_WAIT_MS, config.AEGIS_DEADLINE_MS)
        RT.calibration = {ch: Calibration.load(config.MODEL_DIR, ch) for ch in ("clean", "telephone")}
        RT.risk_policy = RiskPolicy.load(config.MODEL_DIR)
        RT.store = Store("", "")
        RT.hub = Hub()
    RT.aegis.set_policy(policy)
    RT.telemetry.reset()


def gpu_processes() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


async def run_config(policy: str, calls: list[tuple[str, np.ndarray]]) -> dict:
    import torch

    from app.telephony.sim import run_in_process

    setup(policy)
    torch.cuda.reset_peak_memory_stats()
    q = RT.hub.subscribe("bench")
    util, mem = [], []
    stop = False

    async def sample():
        while not stop:
            g = RT.telemetry.gpu.latest if RT.telemetry.gpu else None
            if g:
                util.append(g["util_pct"])
                mem.append(g["mem_used_mb"])
            await asyncio.sleep(0.25)

    async def one(i, label, audio):
        await asyncio.sleep(i * 0.5 / max(1, len(calls)))  # stagger within one hop
        return await run_in_process(audio, {"user_id": "bench", "caller": f"load-{i}", "consent": "simulated",
                                            "scenario": label})

    sampler = asyncio.create_task(sample())
    t0 = time.monotonic()
    sids = await asyncio.gather(*[one(i, lab, a) for i, (lab, a) in enumerate(calls)])
    wall = time.monotonic() - t0
    stop = True
    await sampler
    snap = RT.telemetry.snapshot()
    ended, alerts = {}, {}
    while not q.empty():
        e = q.get_nowait()
        if e["type"] == "call.ended":
            ended[e["session_id"]] = e
        elif e["type"] == "call.alert":
            alerts.setdefault(e["session_id"], e["t_ms"])
    RT.hub.unsubscribe("bench", q)
    labels = dict(zip(sids, [lab for lab, _ in calls]))
    outcome = {}
    for lab in ("genuine", "synthetic"):
        ids = [s for s, l in labels.items() if l == lab]
        states = [ended[s]["final_state"] for s in ids if s in ended]
        outcome[lab] = {"calls": len(ids), "suspicious": states.count("suspicious"),
                        "no_suspicious_evidence": states.count("no_suspicious_evidence"),
                        "inconclusive": states.count("inconclusive"),
                        "analysis_unavailable": states.count("analysis_unavailable"),
                        "first_alert_ms_median": float(np.median([alerts[s] for s in ids if s in alerts]))
                        if any(s in alerts for s in ids) else None}
    a = snap["actions"]
    hops = sum(a.values())
    return {"policy": policy, "calls": len(calls), "wall_s": round(wall, 1),
            "latency_p50_ms": snap["latency_ms"]["p50"], "latency_p95_ms": snap["latency_ms"]["p95"],
            "latency_p99_ms": snap["latency_ms"]["p99"], "compute_p50_ms": snap["compute_ms"]["p50"],
            "batch_mean": snap["batch_size"]["mean"], "batch_max": snap["batch_size"]["max"],
            "hops": hops, "inferred": a["infer"] + a["escalate"], "escalated": a["escalate"],
            "skipped_silence": a["skip_silence"], "skipped_clear": a["skip_clear"], "shed": a["shed"],
            "compute_fraction": round((a["infer"] + a["escalate"]) / hops, 3) if hops else None,
            "gpu_util_mean_pct": round(float(np.mean(util)), 1) if util else None,
            "gpu_mem_peak_mb": max(mem) if mem else None,
            "torch_peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20),
            "genuine": outcome["genuine"], "synthetic": outcome["synthetic"]}


def build_calls(n: int, seconds: float) -> list[tuple[str, np.ndarray]]:
    from app.telephony.sim import scenario_audio

    out = []
    for i in range(n):
        lab = "synthetic" if i % 2 else "genuine"
        out.append((lab, scenario_audio(lab, seed=1000 + i, target_s=seconds)[: int(seconds * 16000)]))
    return out


async def main(a):
    import torch

    pool = build_calls(max(a.calls), a.seconds)
    base = gpu_processes()
    rows = []
    for n in a.calls:
        for pol in a.policies:
            r = await run_config(pol, pool[:n])
            print(json.dumps({k: r[k] for k in ("policy", "calls", "latency_p50_ms", "latency_p95_ms", "inferred",
                                                 "hops", "shed", "gpu_util_mean_pct")}), flush=True)
            rows.append(r)
    out = {"created": datetime.now(timezone.utc).isoformat(), "kind": "load_test",
           "description": ("Concurrent simulated callers (alternating genuine LibriSpeech / synthetic validation or "
                           "Grok audio) streamed through the production Twilio path at real-time pace."),
           "machine": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "python": platform.python_version(),
                       "backend": RT.aegis.backend.name, "max_batch": RT.aegis.max_batch,
                       "max_wait_ms": RT.aegis.max_wait * 1000, "deadline_ms": RT.aegis.deadline * 1000},
           "call_seconds": a.seconds, "gpu_baseline": base, "results": rows}
    p = config.OUTPUT_DIR / "benchmarks" / "load_test.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, nargs="+", default=[1, 10, 20, 30, 40])
    ap.add_argument("--policies", nargs="+", default=["fifo", "dynamic", "edf", "adaptive"])
    ap.add_argument("--seconds", type=float, default=20.0)
    asyncio.run(main(ap.parse_args()))
