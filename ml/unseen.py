#!/usr/bin/env python3
"""Summarise an unseen-generator experiment: a model trained with --exclude-generators vs the reference model
trained on all generators, on the same validation set.

    python ml/unseen.py --exp models/xlsr12_unseen --ref models/xlsr12_v1
Writes outputs/results/unseen_<exp name>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
from common import manifest, save_json  # noqa: E402
from evaluate_mindcf import fast_mindcf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--ref", required=True)
    a = ap.parse_args()
    exp, ref = ROOT / a.exp, ROOT / a.ref
    excluded = json.loads((exp / "meta.json").read_text())["args"]["exclude_generators"]
    m = manifest("val")
    y = m.y.to_numpy()
    pe = 1 / (1 + np.exp(-np.load(exp / "val_logits.npy")))
    pr = 1 / (1 + np.exp(-np.load(ref / "val_logits.npy")))
    bona = y == 0
    rows = []
    for g in sorted(m[m.y == 1].generator.unique()):
        sel = bona | (m.generator == g).to_numpy()
        rows.append({"generator": g, "seen_in_training": g not in excluded,
                     "minDCF_model_without_it": round(fast_mindcf(y[sel], pe[sel])[0], 4) if g in excluded else None,
                     "minDCF_model_trained_on_all": round(fast_mindcf(y[sel], pr[sel])[0], 4),
                     "minDCF_exp_model": round(fast_mindcf(y[sel], pe[sel])[0], 4)})
    seen = ~m.generator.isin(excluded).to_numpy()
    out = {"experiment": a.exp, "excluded_generators": excluded, "rows": [r for r in rows if not r["seen_in_training"]],
           "all_rows": rows,
           "pooled_seen_only_exp": round(fast_mindcf(y[seen], pe[seen])[0], 4),
           "pooled_unseen_only_exp": round(fast_mindcf(y[bona | ~seen], pe[bona | ~seen])[0], 4)}
    save_json(out, ROOT / f"outputs/results/unseen_{exp.name}.json")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
