#!/usr/bin/env python3
"""(1) Score real Grok Voice clips (data/grok_samples + api fixture) with the selected detector: an unseen,
commercial, real-time generator never present in training.  (2) Measure inference latency (GPU and CPU).

    python ml/score_ood.py
Writes outputs/results/grok_ood.json and outputs/results/latency.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "ml"))
from common import save_json  # noqa: E402


def main():
    import torch

    from app.audio.ingest import load_audio
    from app.detectors.primary import PrimaryDetector

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    det = PrimaryDetector(ROOT / "models/selected", dev)
    files = sorted((ROOT / "data/grok_samples").glob("*.wav")) + [ROOT / "api/app/fixtures/grok_fixture.wav"]
    rows = []
    for f in files:
        x, sr, _ = load_audio(f)
        r = det.score(x)
        w = det.score_windows(x)
        side = f.with_suffix(".json")
        text = json.loads(side.read_text()).get("text") if side.exists() else None
        rows.append({"file": f.name, "source": "grok_voice", "duration_s": round(len(x) / 16000, 2),
                     "synthetic_probability": round(r.synthetic_probability, 4),
                     "flagged_at_threshold": bool(r.synthetic_probability >= det.threshold),
                     "min_window_p": round(min(v["synthetic_probability"] for v in w), 4),
                     "max_window_p": round(max(v["synthetic_probability"] for v in w), 4), "text": text})
        print(rows[-1], flush=True)
    save_json({"threshold": det.threshold, "rows": rows,
               "flagged": f"{sum(r['flagged_at_threshold'] for r in rows)}/{len(rows)}"},
              ROOT / "outputs/results/grok_ood.json")

    lat = {}
    rng = np.random.default_rng(0)
    for name, d in [("gpu_fp16", det)] + ([("cpu_fp32", PrimaryDetector(ROOT / "models/selected", "cpu"))]
                                          if dev == "cuda" else []):
        for secs in (2.0, 4.0):
            x = (0.1 * rng.standard_normal(int(secs * 16000))).astype(np.float32)
            d.logits([x])
            ts = []
            for _ in range(10):
                t = time.perf_counter()
                d.logits([x])
                ts.append((time.perf_counter() - t) * 1000)
            lat[f"{name}_{secs:.0f}s_clip_ms_median"] = round(float(np.median(ts)), 1)
    save_json(lat, ROOT / "outputs/results/latency.json")
    print(lat)


if __name__ == "__main__":
    main()
