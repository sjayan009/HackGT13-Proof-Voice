#!/usr/bin/env python3
"""Calibrate a trained detector on validation and (optionally) freeze it as models/selected.

* Re-scores validation through the SERVING code path (api PrimaryDetector) on the standard test-like crops.
* Platt scaling p = sigmoid(a * logit + b), fitted with class-balanced weights (i.e. assumes equal priors; the
  held-out prior of ~70 % bona fide is NOT baked in). Monotone => minDCF is unchanged by calibration.
* decision_threshold = p threshold minimising validation minDCF (used for UI status only; the TSV carries raw p).

    python ml/calibrate.py --model-dir models/xlsr12_v1 [--select]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "api"))
from audio_cache import Cache  # noqa: E402
from common import manifest, save_json, subset_report, testlike_crop  # noqa: E402
from evaluate_mindcf import fast_mindcf, full_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--select", action="store_true", help="copy to models/selected")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    import torch
    from sklearn.linear_model import LogisticRegression

    from app.detectors.primary import PrimaryDetector

    md = ROOT / a.model_dir
    (md / "calibration.json").unlink(missing_ok=True)
    det = PrimaryDetector(md, "cuda" if torch.cuda.is_available() else "cpu")
    m, c = manifest("val"), Cache("val")
    clips = [testlike_crop(c.get(i), i, 0) for i in range(len(c))]
    lg = det.logits(clips, bs=8).astype(np.float64)
    y = m.y.to_numpy()
    w = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(lg[:, None], y, sample_weight=w)
    a_, b_ = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    p = 1 / (1 + np.exp(-(a_ * lg + b_)))
    md_, thr = fast_mindcf(y, p)
    rep = full_report(y, p)
    cal = {"a": a_, "b": b_, "decision_threshold": float(thr), "val_minDCF": md_,
           "fitted_on": "validation manifest, test-like crops, class-balanced (equal priors)",
           "val_ECE_10bin": float(ece(y, p))}
    (md / "calibration.json").write_text(json.dumps(cal, indent=2))
    np.save(md / "val_logits_serving.npy", lg)
    sub = subset_report(m, p)
    save_json(sub | {"calibration": cal}, md / "val_report_serving.json")
    print(json.dumps({"calibration": cal, "pooled": rep, "per_gen": sub["per_generator_minDCF"],
                      "lj": sub.get("lj_bonafide_vs_lj_voice_spoofs_minDCF"),
                      "libri": sub.get("libri_bonafide_vs_cloned_spoofs_minDCF")}, indent=1))
    if a.select:
        meta = json.loads((md / "meta.json").read_text())
        meta["name"] = a.name or f"ProofVoice-{Path(a.model_dir).name}"
        meta["version"] = Path(a.model_dir).name
        (md / "meta.json").write_text(json.dumps(meta, indent=2))
        dst = ROOT / "models/selected"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(md, dst)
        print("selected ->", dst)


def ece(y, p, n=10):
    bins = np.clip((p * n).astype(int), 0, n - 1)
    return sum(abs(y[bins == k].mean() - p[bins == k].mean()) * (bins == k).mean() for k in range(n) if (bins == k).any())


if __name__ == "__main__":
    main()
