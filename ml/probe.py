#!/usr/bin/env python3
"""Linear probes on frozen SSL embeddings: per-layer sweep, then leave-one-generator-out (LOGO).

    python ml/probe.py --tag wav2vec2-xls-r-300m
Writes outputs/results/probe_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
from common import manifest, save_json, subset_report  # noqa: E402
from evaluate_mindcf import fast_mindcf  # noqa: E402


def fit_lr(X, y, C=0.5):
    sc = StandardScaler().fit(X)
    w = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))  # class-balanced
    lr = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X), y, sample_weight=w)
    return sc, lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layers", default="all")
    a = ap.parse_args()
    E = ROOT / "outputs/embeddings"
    Xtr = np.load(E / f"{a.tag}_train.npy").astype(np.float32)
    Xva = np.load(E / f"{a.tag}_val.npy").astype(np.float32)
    mtr, mva = manifest("train"), manifest("val")
    ytr, yva = mtr.y.to_numpy(), mva.y.to_numpy()
    L = Xtr.shape[1]
    layers = range(L) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    sweep = {}
    for l in layers:
        sc, lr = fit_lr(Xtr[:, l], ytr)
        p = lr.predict_proba(sc.transform(Xva[:, l]))[:, 1]
        sweep[l] = round(fast_mindcf(yva, p)[0], 4)
        print("layer", l, sweep[l], flush=True)
    best = min(sweep, key=sweep.get)
    sc, lr = fit_lr(Xtr[:, best], ytr)
    p = lr.predict_proba(sc.transform(Xva[:, best]))[:, 1]
    np.save(ROOT / f"outputs/results/val_p_probe_{a.tag}.npy", p)
    rep = {"tag": a.tag, "layer_sweep_minDCF": sweep, "best_layer": best, "val": subset_report(mva, p)}
    # LOGO: drop generator g from training entirely, evaluate on val bona fide + val spoofs of g.
    logo = {}
    for g in sorted(mtr[mtr.y == 1].generator.unique()):
        keep = (mtr.generator != g).to_numpy()
        sc_g, lr_g = fit_lr(Xtr[keep, best], ytr[keep])
        sel = ((mva.y == 0) | (mva.generator == g)).to_numpy()
        pg = lr_g.predict_proba(sc_g.transform(Xva[sel, best]))[:, 1]
        logo[g] = round(fast_mindcf(yva[sel], pg)[0], 4)
        print("LOGO", g, logo[g], flush=True)
    rep["logo_unseen_generator_minDCF"] = logo
    rep["logo_mean"] = round(float(np.mean(list(logo.values()))), 4)
    save_json(rep, ROOT / f"outputs/results/probe_{a.tag}.json")
    print(json.dumps({k: rep[k] for k in ["best_layer", "logo_mean"]} | {"val_pooled": rep["val"]["pooled"]}, indent=1))


if __name__ == "__main__":
    main()
