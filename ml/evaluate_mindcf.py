#!/usr/bin/env python3
"""Official HEARSAY minDCF wrapper.

Authority: the organizer-provided `HackGTMinDCF.zip`, which is the ASVspoof 5
evaluation package with Track-1 costs modified by the organizers to

    Pspoof = 0.5, Cmiss = 1, Cfa = 4

so that, after normalisation,  minDCF = min_t [ FRR_bonafide(t) + 4 * FAR_spoof(t) ].

SCORE ORIENTATION (important)
-----------------------------
The organizer code treats bona fide as the *target* class: a HIGHER cm-score
means MORE BONA FIDE.  ProofVoice internally uses p_synthetic (1.0 = synthetic),
which is what the HEARSAY instructions ask for in the TSV.  This module therefore
exposes both:

  * `official_mindcf(bona_scores, spoof_scores)`  – calls organizer code verbatim
    on bona-fide-oriented scores.
  * `mindcf_from_psynth(y_spoof, p_synth)` – converts p_synth -> (1 - p_synth)
    before calling the official code.

CLI (mirrors organizer `evaluation.py --m t1`):

    python ml/evaluate_mindcf.py --cm scores.tsv --cm_keys key.tsv [--orientation bonafide_high|synthetic_high]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DIR = ROOT / "data/hearsay/scoring/HackGTMinDCF/HackGTMinDCF/asvspoof5/evaluation-package"
# Vendored verbatim copy (so Docker/tests work without the organizer zip).
VENDORED_DIR = ROOT / "ml/official_scoring"

PSPOOF, CMISS, CFA = 0.5, 1.0, 4.0


def _official_modules():
    d = OFFICIAL_DIR if (OFFICIAL_DIR / "calculate_modules.py").exists() else VENDORED_DIR
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    import calculate_modules  # type: ignore  # noqa: E402

    return calculate_modules


def official_metrics(bona: np.ndarray, spoof: np.ndarray) -> dict:
    """Exactly the computation in organizer calculate_metrics.calculate_minDCF_EER_CLLR_actDCF
    (without its file/`cat` side effects). Scores: higher = bona fide."""
    cm = _official_modules()
    bona = np.asarray(bona, dtype=np.float64)
    spoof = np.asarray(spoof, dtype=np.float64)
    with contextlib.redirect_stdout(io.StringIO()):
        eer, frr, far, thr, _ = cm.compute_eer(bona, spoof)
        cllr = cm.calculate_CLLR(bona, spoof)
        mindcf, mthr = cm.compute_mindcf(frr, far, thr, PSPOOF, CMISS, CFA)
        actdcf, _ = cm.compute_actDCF(bona, spoof, PSPOOF, CMISS, CFA)
    return {"minDCF": float(mindcf), "EER": float(eer), "CLLR": float(cllr),
            "actDCF": float(actdcf), "minDCF_threshold": float(mthr)}


def official_mindcf(bona: np.ndarray, spoof: np.ndarray) -> float:
    return official_metrics(bona, spoof)["minDCF"]


def fast_mindcf(y_spoof: np.ndarray, p_synth: np.ndarray) -> tuple[float, float]:
    """Vectorised reimplementation (validated against official in tests).
    Returns (minDCF, p_synth threshold: flag synthetic if p >= thr)."""
    y = np.asarray(y_spoof).astype(bool)
    s = 1.0 - np.asarray(p_synth, dtype=np.float64)  # bona-fide-oriented
    bona, spoof = s[~y], s[y]
    cm = _official_modules()
    frr, far, thr = cm.compute_det_curve(bona, spoof)
    cdet = CMISS * frr * (1 - PSPOOF) + CFA * far * PSPOOF
    i = int(np.argmin(cdet))
    return float(cdet[i] / min(CMISS * (1 - PSPOOF), CFA * PSPOOF)), float(1.0 - thr[i])


def mindcf_from_psynth(y_spoof, p_synth) -> dict:
    y = np.asarray(y_spoof).astype(bool)
    s = 1.0 - np.asarray(p_synth, dtype=np.float64)
    return official_metrics(s[~y], s[y])


def full_report(y_spoof, p_synth) -> dict:
    """minDCF (official) + secondary metrics used for model selection."""
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

    y = np.asarray(y_spoof).astype(int)
    p = np.clip(np.asarray(p_synth, dtype=np.float64), 1e-6, 1 - 1e-6)
    r = mindcf_from_psynth(y, p)
    _, thr = fast_mindcf(y, p)
    pred = p >= thr
    r.update({
        "AUC": float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan"),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "FRR_bonafide_at_minDCF": float(np.mean(pred[y == 0])) if (y == 0).any() else float("nan"),
        "FAR_spoof_at_minDCF": float(np.mean(~pred[y == 1])) if (y == 1).any() else float("nan"),
        "p_synth_threshold_at_minDCF": thr,
        "n_bonafide": int((y == 0).sum()), "n_spoof": int((y == 1).sum()),
    })
    return r


def _load(path_scores: str, path_keys: str):
    import pandas as pd

    s = pd.read_csv(path_scores, sep="\t")
    k = pd.read_csv(path_keys, sep="\t")
    df = s.merge(k, on="filename", how="inner", validate="one_to_one")
    if len(df) != len(k):
        raise SystemExit(f"score/key mismatch: {len(df)} merged vs {len(k)} keys")
    return df["cm-score"].to_numpy(float), df["cm-label"].to_numpy()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cm", required=True)
    ap.add_argument("--cm_keys", required=True)
    ap.add_argument("--orientation", choices=["bonafide_high", "synthetic_high"], default="bonafide_high",
                    help="bonafide_high = organizer convention (default); synthetic_high = HEARSAY p_synthetic")
    a = ap.parse_args(argv)
    scores, labels = _load(a.cm, a.cm_keys)
    if a.orientation == "synthetic_high":
        scores = 1.0 - scores
    r = official_metrics(scores[labels == "bonafide"], scores[labels == "spoof"])
    print("# Track 1 Result (HEARSAY costs Pspoof=0.5 Cmiss=1 Cfa=4):\n")
    print("-eval_mindcf: {minDCF:.5f}\n-eval_eer (%): {e:.3f}\n-eval_cllr (bits): {CLLR:.5f}\n-eval_actDCF: {actDCF:.5f}".format(
        e=r["EER"] * 100, **r))


if __name__ == "__main__":
    main()
