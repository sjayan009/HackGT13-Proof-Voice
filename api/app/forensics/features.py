"""Flat, fixed-key feature vector across all sample-derived forensic branches.

`metadata` is deliberately excluded from `all_branch_features` -- its features
depend on the original file path/container, not on the decoded sample array, so
it is not something the ML fusion benchmark can compute from `(audio, sr)` alone.
Fetch it separately via `metadata.inspect_file` + `metadata.analyze_metadata` when
a file path is available.
"""
from __future__ import annotations

import math

import numpy as np

from . import compression, prosody, quality, splice, spectral

_BRANCH_MODULES = {
    "spectral": spectral,
    "prosody": prosody,
    "compression": compression,
    "splice": splice,
    "quality": quality,
}


def all_branch_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Concatenate `extract_features` from spectral/prosody/compression/splice/quality.

    Keys are prefixed with the branch name, e.g. `spectral.flatness_mean`.
    NaN/inf-safe: any non-finite value is replaced with 0.0 and a matching
    `<key>_missing` indicator (1.0) is added so downstream fusion can tell a
    real zero from a missing value.
    """
    out: dict[str, float] = {}
    for name, module in _BRANCH_MODULES.items():
        try:
            feats = module.extract_features(audio, sr)
        except Exception:
            feats = {}
        for k, v in feats.items():
            key = f"{name}.{k}"
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = float("nan")
            if not math.isfinite(fv):
                out[key] = 0.0
                out[f"{key}_missing"] = 1.0
            else:
                out[key] = fv
    return out
