"""Per-window latency of the screening student vs the main model (PyTorch FP16), interleaved so both see the
same GPU conditions. The ratio feeds benchmarks/cascade_replay.py --cost-ratio.

    .venv/Scripts/python.exe benchmarks/latency/screen_ratio.py --student models/exp_student_l4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "api"))


def main(a):
    import torch

    from app.detectors.primary import PrimaryDetector

    models = {"main": PrimaryDetector(ROOT / "models" / "selected", "cuda"),
              "student": PrimaryDetector(ROOT / a.student, "cuda")}
    rng = np.random.default_rng(0)
    rows = []
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    for bs in a.batch:
        clips = [rng.standard_normal(48000).astype(np.float32) * 0.1 for _ in range(bs)]
        times = {k: [] for k in models}
        for m in models.values():  # warm-up
            for _ in range(5):
                m.logits(clips, bs=bs)
        for _ in range(a.iters):
            for k, m in models.items():
                torch.cuda.synchronize()
                t = time.perf_counter()
                m.logits(clips, bs=bs)
                torch.cuda.synchronize()
                times[k].append((time.perf_counter() - t) * 1000)
        med = {k: float(np.median(v)) for k, v in times.items()}
        rows.append({"batch": bs, "main_p50_ms": round(med["main"], 2), "student_p50_ms": round(med["student"], 2),
                     "ratio": round(med["student"] / med["main"], 3)})
        print(rows[-1], flush=True)
    out = {"created": datetime.now(timezone.utc).isoformat(), "kind": "screen_ratio", "student": a.student,
           "other_gpu_processes": gpu or "none", "results": rows,
           "note": "Interleaved measurement; absolute times are inflated if other GPU work was running, the ratio less so."}
    (ROOT / "outputs" / "benchmarks" / "screen_ratio.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="models/exp_student_l4")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--iters", type=int, default=40)
    main(ap.parse_args())
