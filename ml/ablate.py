#!/usr/bin/env python3
"""Forensic-branch usefulness + fusion ablation, judged by official minDCF.

1. Standalone: each hand-crafted branch (features -> logistic regression trained on TRAIN) evaluated on VAL.
2. Fusion: primary-detector logit + branch features, stacked with logistic regression using 5-fold GROUPED
   cross-validation on VAL (the primary detector was trained on TRAIN, so its train logits are optimistic and
   cannot be used to fit a fusion).  Out-of-fold predictions -> minDCF.  Fusion is kept only if it beats primary.

    python ml/ablate.py --primary models/xlsr12_v1
Writes outputs/results/fusion.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
from common import manifest, save_json  # noqa: E402
from evaluate_mindcf import fast_mindcf, full_report  # noqa: E402

BRANCHES = ["spectral", "prosody", "compression", "splice", "quality"]


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced"))


def oof(X, y, groups, n=5):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n).split(X, y, groups):
        p[te] = lr().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True)
    a = ap.parse_args()
    mtr, mva = manifest("train"), manifest("val")
    ytr, yva = mtr.y.to_numpy(), mva.y.to_numpy()
    Btr = pd.read_parquet(ROOT / "outputs/embeddings/branch_train.parquet")
    Bva = pd.read_parquet(ROOT / "outputs/embeddings/branch_val.parquet")
    Btr, Bva = Btr.fillna(0).replace([np.inf, -np.inf], 0), Bva.fillna(0).replace([np.inf, -np.inf], 0)
    groups = mva.group.to_numpy()
    res = {"standalone_train_to_val": {}, "fusion_oof_on_val": {}}

    for b in BRANCHES + ["all_branches"]:
        cols = [c for c in Btr.columns if b == "all_branches" or c.startswith(b + ".")]
        m = lr().fit(Btr[cols].to_numpy(), ytr)
        p = m.predict_proba(Bva[cols].to_numpy())[:, 1]
        r = full_report(yva, p)
        res["standalone_train_to_val"][b] = {"n_features": len(cols), "minDCF": round(r["minDCF"], 4),
                                             "AUC": round(r["AUC"], 4), "EER": round(r["EER"], 4)}
        print("standalone", b, res["standalone_train_to_val"][b], flush=True)

    lg = np.load(ROOT / a.primary / "val_logits.npy")
    base = fast_mindcf(yva, 1 / (1 + np.exp(-lg)))[0]
    p0 = oof(lg[:, None], yva, groups)
    res["fusion_oof_on_val"]["primary_only"] = round(fast_mindcf(yva, p0)[0], 4)
    res["primary_raw_minDCF"] = round(base, 4)
    for b in BRANCHES + ["all_branches"]:
        cols = [c for c in Bva.columns if b == "all_branches" or c.startswith(b + ".")]
        X = np.column_stack([lg, Bva[cols].to_numpy()])
        res["fusion_oof_on_val"][f"primary+{b}"] = round(fast_mindcf(yva, oof(X, yva, groups))[0], 4)
        print("fusion", b, res["fusion_oof_on_val"][f"primary+{b}"], flush=True)
    best = min(res["fusion_oof_on_val"], key=res["fusion_oof_on_val"].get)
    res["best_config"] = best
    res["decision"] = ("keep primary only: no fusion beat primary_only out-of-fold"
                       if best == "primary_only" or res["fusion_oof_on_val"][best] >= res["fusion_oof_on_val"]["primary_only"] - 0.005
                       else f"fusion {best} improves out-of-fold minDCF")
    save_json(res, ROOT / "outputs/results/fusion.json")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
