"""Shared ML helpers: manifests, test-like cropping, per-subset evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
# Held-out clips: min 3.02 s, median 3.41 s, mean 3.69 s, p90 4.57 s.  Validation uses crops drawn
# from that range so model selection reflects the held-out duration regime.
TESTLIKE_MIN_S, TESTLIKE_MAX_S = 3.0, 4.6


def manifest(split: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / f"outputs/manifests/{split}.csv", dtype={"sentence": str})


def testlike_crop(x: np.ndarray, i: int, seed: int = 0) -> np.ndarray:
    """Deterministic per-clip crop with test-like duration (seeded by clip index)."""
    rng = np.random.default_rng(seed * 1_000_003 + i)
    n = int(rng.uniform(TESTLIKE_MIN_S, TESTLIKE_MAX_S) * SR)
    if len(x) <= n:
        return x
    o = int(rng.integers(0, len(x) - n))
    return x[o:o + n]


def subset_report(m: pd.DataFrame, p: np.ndarray) -> dict:
    """Pooled metrics plus per-generator minDCF (each generator vs all bona fide) and LJ-only views."""
    import sys
    sys.path.insert(0, str(ROOT / "ml"))
    from evaluate_mindcf import fast_mindcf, full_report

    y = m.y.to_numpy()
    out = {"pooled": full_report(y, p)}
    bona = y == 0
    per = {}
    for g in sorted(m[m.y == 1].generator.unique()):
        sel = bona | (m.generator.to_numpy() == g)
        per[g] = round(fast_mindcf(y[sel], p[sel])[0], 4)
    out["per_generator_minDCF"] = per
    lj = (m.generator == "bonafide_lj").to_numpy()
    single = m.speaker.eq("single").to_numpy() & (y == 1)
    if lj.any() and single.any():
        sel = lj | single
        out["lj_bonafide_vs_lj_voice_spoofs_minDCF"] = round(fast_mindcf(y[sel], p[sel])[0], 4)
    libri = m.source.eq("librispeech").to_numpy()
    multi = (~m.speaker.eq("single")).to_numpy() & (y == 1)
    if libri.any() and multi.any():
        sel = libri | multi
        out["libri_bonafide_vs_cloned_spoofs_minDCF"] = round(fast_mindcf(y[sel], p[sel])[0], 4)
    out["bonafide_p_by_source"] = {s: [round(float(np.quantile(p[(m.source == s).to_numpy() & bona], q)), 4)
                                       for q in (.5, .9, .99)] for s in m[m.y == 0].source.unique()}
    return out


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
