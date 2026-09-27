#!/usr/bin/env python3
"""Re-score every trained model's validation logits under the costs NSA announced for final scoring.

NSA (HackGT 13 Discord, PuzzleMaster): evaluation uses calculate_metrics.py from the ASVspoof5
evaluation package with **Pspoof = 0.3 and Cfa = 4** (Cmiss = 1). Our model selection was run with
Pspoof = 0.5 (the value in HackGTMinDCF.zip). minDCF chooses its own threshold, so the submitted TSV
does not change; only the reported number does. Both are written here, computed with the organizer's
own `compute_det_curve` / `compute_mindcf` / `compute_eer`.

    .venv/Scripts/python ml/rescore_nsa_settings.py      ->  outputs/results/mindcf_nsa_settings.json
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_mindcf as em

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ("xlsr12_v3", "val_logits_serving.npy", "selected (submitted TSV)"),
    ("xlsr12_v4_codec_hard", "val_logits_serving.npy", "v4 codec/hard augmentation (not selected)"),
    ("xlsr12_v2_lj30", "val_logits_serving.npy", ""),
    ("xlsr12_v1", "val_logits_serving.npy", ""),
    ("aasist_v1", "val_logits.npy", "AASIST fine-tuned"),
]
SETTINGS = {"nsa_final (Pspoof=0.3, Cmiss=1, Cfa=4)": 0.3, "HackGTMinDCF.zip (Pspoof=0.5, Cmiss=1, Cfa=4)": 0.5}


def main() -> None:
    cm = em._official_modules()
    y = pd.read_csv(ROOT / "outputs/manifests/val.csv").y.to_numpy().astype(bool)
    rows = []
    for name, fname, note in MODELS:
        f = ROOT / "models" / name / fname
        if not f.exists():
            continue
        s = -np.load(f).astype(np.float64)  # logits are spoof-vs-bona; organizer code wants higher = bona fide
        bona, spoof = s[~y], s[y]
        row = {"model": name, "note": note, "n_bonafide": int((~y).sum()), "n_spoof": int(y.sum())}
        with contextlib.redirect_stdout(io.StringIO()):
            eer, frr, far, thr, _ = cm.compute_eer(bona, spoof)
            for label, ps in SETTINGS.items():
                mindcf, _ = cm.compute_mindcf(frr, far, thr, ps, 1.0, 4.0)
                row[f"minDCF {label}"] = round(float(mindcf), 4)
        row["EER_percent"] = round(float(eer) * 100, 2)
        rows.append(row)
        print(f"{name:22s} " + "  ".join(f"{k.split(' ')[0]}={row[f'minDCF {k}']:.4f}" for k in SETTINGS) + f"  EER={row['EER_percent']}%")
    out = ROOT / "outputs/results/mindcf_nsa_settings.json"
    out.write_text(json.dumps({"validation_split": "outputs/manifests/val.csv (group-disjoint)", "rows": rows}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
