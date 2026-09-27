#!/usr/bin/env python3
"""Expose subgroup failures hidden by the pooled validation minDCF.

Uses only frozen validation logits; it never reads held-out labels or changes the
selected model. Run after calibrate.py:

    python ml/audit_generalization.py --model-dir models/selected
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import manifest, save_json
from evaluate_mindcf import fast_mindcf

ROOT = Path(__file__).resolve().parents[1]


def row(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    min_dcf, _ = fast_mindcf(y, p)
    bona, spoof = y == 0, y == 1
    return {
        "n_bonafide": int(bona.sum()),
        "n_spoof": int(spoof.sum()),
        "minDCF_with_reoptimized_threshold": round(min_dcf, 4),
        "false_alarm_rate_at_global_threshold": round(float((p[bona] >= threshold).mean()), 4),
        "miss_rate_at_global_threshold": round(float((p[spoof] < threshold).mean()), 4),
    }


def audit(model_dir: Path) -> dict:
    m = manifest("val")
    y = m.y.to_numpy()
    cal = json.loads((model_dir / "calibration.json").read_text())
    logits = np.load(model_dir / "val_logits_serving.npy")
    if len(logits) != len(m):
        raise ValueError("validation logits and manifest have different lengths")
    p = 1 / (1 + np.exp(-(cal["a"] * logits + cal["b"])))
    threshold = float(cal["decision_threshold"])
    bona = y == 0
    out = {
        "model_dir": str(model_dir),
        "global_threshold": threshold,
        "pooled": row(y, p, threshold),
        "spoof_speaker_subgroups": {},
        "bona_fide_false_alarms_by_source": {},
        "bona_fide_false_alarms_by_speaker": {},
    }
    for (generator, speaker), group in m[m.y == 1].groupby(["generator", "speaker"]):
        sel = bona.copy()
        sel[group.index.to_numpy()] = True
        out["spoof_speaker_subgroups"][f"{generator}/{speaker}"] = row(y[sel], p[sel], threshold)
    for source, group in m[m.y == 0].groupby("generator"):
        idx = group.index.to_numpy()
        out["bona_fide_false_alarms_by_source"][source] = {
            "n": int(len(idx)), "false_alarms": int((p[idx] >= threshold).sum()),
            "false_alarm_rate": round(float((p[idx] >= threshold).mean()), 4),
        }
    for speaker, group in m[m.y == 0].groupby("speaker"):
        idx = group.index.to_numpy()
        out["bona_fide_false_alarms_by_speaker"][speaker] = {
            "n": int(len(idx)), "false_alarms": int((p[idx] >= threshold).sum()),
            "false_alarm_rate": round(float((p[idx] >= threshold).mean()), 4),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/selected")
    ap.add_argument("--output", default="outputs/results/generalization_audit.json")
    args = ap.parse_args()
    result = audit(ROOT / args.model_dir)
    save_json(result, ROOT / args.output)
    print(json.dumps({"pooled": result["pooled"],
                      "hardest_groups": sorted(result["spoof_speaker_subgroups"].items(),
                                              key=lambda item: item[1]["minDCF_with_reoptimized_threshold"],
                                              reverse=True)[:8],
                      "bona_fide_source": result["bona_fide_false_alarms_by_source"]}, indent=2))


if __name__ == "__main__":
    main()
