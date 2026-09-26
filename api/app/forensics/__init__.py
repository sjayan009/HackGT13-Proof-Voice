"""CPU-only, deterministic forensic analysis branches for ProofVoice.

Each branch module exposes:
  - `analyze(audio: np.ndarray, sr: int, **ctx) -> dict` matching the
    `evidence.<technique>` schema in `api/CONTRACT.md`
    ({"summary", "suspicion", "features", "flags", "used_in_score"}).
  - `extract_features(audio: np.ndarray, sr: int) -> dict[str, float]`, a flat
    fixed-key numeric feature vector for later fusion benchmarking.

`metadata.py` additionally exposes `inspect_file(path)` and `analyze_metadata(info)`
since metadata evidence needs the original file, not just decoded samples.

`features.all_branch_features(audio, sr)` concatenates the sample-derived branches'
feature vectors with `<branch>.` prefixed keys.
"""
from . import compression, metadata, prosody, quality, spectral, splice
from .features import all_branch_features

__all__ = [
    "compression", "metadata", "prosody", "quality", "spectral", "splice",
    "all_branch_features",
]
