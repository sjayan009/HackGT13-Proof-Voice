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

    # Weighted-logit fusion: s = primary_logit + w * branch_logit, branch LR fitted on TRAIN (branch features are
    # not overfit, unlike the primary's train logits). w chosen on 4 val folds, evaluated on the 5th; the reported
    # number is the mean per-fold minDCF (no cross-fold score mixing). primary_only uses the identical folds.
    lg = np.load(ROOT / a.primary / "val_logits.npy")
    res["primary_raw_minDCF"] = round(fast_mindcf(yva, 1 / (1 + np.exp(-lg)))[0], 4)
    from sklearn.model_selection import StratifiedGroupKFold
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(lg, yva, groups))
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    W = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

    def perfold(z_branch):
        vals, ws = [], []
        for tr, te in folds:
            best_w = min(W, key=lambda w: fast_mindcf(yva[tr], sig(lg[tr] + w * z_branch[tr]))[0])
            ws.append(best_w)
            vals.append(fast_mindcf(yva[te], sig(lg[te] + best_w * z_branch[te]))[0])
        return round(float(np.mean(vals)), 4), ws

    res["fusion_oof_on_val"]["primary_only"] = round(float(np.mean([fast_mindcf(yva[te], sig(lg[te]))[0]
                                                                     for _, te in folds])), 4)
    res["fusion_weights_chosen"] = {}
    for b in BRANCHES + ["all_branches"]:
        cols = [c for c in Btr.columns if b == "all_branches" or c.startswith(b + ".")]
        mdl = lr().fit(Btr[cols].to_numpy(), ytr)
        z = mdl.decision_function(Bva[cols].to_numpy())
        v, ws = perfold(z)
        res["fusion_oof_on_val"][f"primary+{b}"] = v
        res["fusion_weights_chosen"][b] = ws
        print("fusion", b, v, ws, flush=True)
    best = min(res["fusion_oof_on_val"], key=res["fusion_oof_on_val"].get)
    res["best_config"] = best
    res["decision"] = ("keep primary only: no fusion beat primary_only out-of-fold"
                       if best == "primary_only" or res["fusion_oof_on_val"][best] >= res["fusion_oof_on_val"]["primary_only"] - 0.005
                       else f"fusion {best} improves out-of-fold minDCF")
    save_json(res, ROOT / "outputs/results/fusion.json")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
