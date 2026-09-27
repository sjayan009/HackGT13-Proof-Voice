#!/usr/bin/env python3
"""Clip-score aggregation for audio longer than the training regime (3.0-4.6 s crops).

On FULL-LENGTH validation clips (up to 14 s), compare minDCF of:
  full    - one forward pass over the whole clip
  mean    - mean logit of 4 s windows (hop 2 s)
  median  - median window logit
  max     - max window logit
  top2    - mean of the two highest window logits
Reported on all clips and on clips longer than 6 s.  Held-out clips are 3.0-13.6 s (90 % < 4.6 s), so this
mostly matters for the ~10 % long held-out clips and for user uploads.

    python ml/aggregation.py --model-dir models/selected
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "api"))
from audio_cache import Cache  # noqa: E402
from common import manifest, save_json  # noqa: E402
from evaluate_mindcf import fast_mindcf  # noqa: E402

SR = 16000
W, H = 4 * SR, 2 * SR


def windows(x):
    if len(x) <= W:
        return [x]
    starts = list(range(0, len(x) - W + 1, H))
    if starts[-1] + W < len(x):
        starts.append(len(x) - W)
    return [x[s:s + W] for s in starts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/selected")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from app.detectors.primary import PrimaryDetector

    det = PrimaryDetector(ROOT / a.model_dir, a.device)
    m, c = manifest("val"), Cache("val")
    rng = np.random.default_rng(5)  # stratified 2,000-clip subsample (full-length clips are memory-heavy on 8 GB)
    yy = m.y.to_numpy()
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(yy == k), 1000, replace=False) for k in (0, 1)]))
    y = yy[idx]
    clips = [c.get(int(i)) for i in idx]
    lens = np.array([len(x) / SR for x in clips])
    full = det.logits(clips, bs=1)
    wl = [det.logits(windows(x), bs=4) for x in clips]
    aggs = {"full": full, "mean": np.array([w.mean() for w in wl]), "median": np.array([np.median(w) for w in wl]),
            "max": np.array([w.max() for w in wl]),
            "top2": np.array([np.sort(w)[-2:].mean() for w in wl])}
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    long_ = lens > 6
    out = {"n": int(len(y)), "n_long_gt6s": int(long_.sum()), "rows": []}
    for k, v in aggs.items():
        out["rows"].append({"aggregation": k, "minDCF_all_full_length": round(fast_mindcf(y, sig(v))[0], 4),
                            "minDCF_clips_gt_6s": round(fast_mindcf(y[long_], sig(v[long_]))[0], 4)})
        print(out["rows"][-1], flush=True)
    save_json(out, ROOT / "outputs/results/aggregation.json")


if __name__ == "__main__":
    main()
