#!/usr/bin/env python3
"""Hand-crafted forensic branch features (spectral / prosody / compression / splice / quality) on the same
test-like crops the detectors see.  Used for standalone-usefulness and fusion ablations.

    python ml/extract_branch_features.py --splits train val
Output: outputs/embeddings/branch_<split>.parquet (one column per feature)
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "api"))


def _chunk(args):
    split, lo, hi, seed = args
    from audio_cache import Cache
    from common import testlike_crop
    from app.forensics.features import all_branch_features

    c = Cache(split)
    return [all_branch_features(testlike_crop(c.get(i), i, seed), 16000) for i in range(lo, hi)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    a = ap.parse_args()
    from audio_cache import Cache

    for split in a.splits:
        n = len(Cache(split))
        seed = 7 if split == "train" else 0  # identical crops to extract_ssl.py
        jobs = [(split, lo, min(n, lo + 500), seed) for lo in range(0, n, 500)]
        with ProcessPoolExecutor(14) as ex:
            rows = [r for part in ex.map(_chunk, jobs) for r in part]
        df = pd.DataFrame(rows).astype(np.float32)
        df.to_parquet(ROOT / f"outputs/embeddings/branch_{split}.parquet")
        print(split, df.shape, flush=True)


if __name__ == "__main__":
    main()
