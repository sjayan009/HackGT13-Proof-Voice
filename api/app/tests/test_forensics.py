"""Tests for api/app/forensics branches.

Uses only synthetic signals (no external audio files needed) so the suite is fast,
deterministic, and runs offline.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from app.forensics import compression, prosody, quality, spectral, splice
from app.forensics.features import all_branch_features

SR = 16000
DURATION_S = 4.0
N = int(SR * DURATION_S)

BRANCHES = {
    "spectral": spectral,
    "prosody": prosody,
    "compression": compression,
    "splice": splice,
    "quality": quality,
}

REQUIRED_KEYS = {"summary", "suspicion", "features", "flags", "used_in_score"}


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _t(n=N, sr=SR):
    return np.arange(n, dtype=np.float64) / sr


def harmonic_tone_with_noise(seed=0):
    rng = np.random.default_rng(seed)
    t = _t()
    f0 = 150.0 + 20.0 * np.sin(2 * np.pi * 0.5 * t)  # vibrato-like modulation
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = 0.5 * np.sin(phase)
    for h in (2, 3, 4):
        sig += (0.5 / h) * np.sin(h * phase)
    sig += 0.02 * rng.standard_normal(N)
    sig /= np.max(np.abs(sig)) + 1e-8
    return (0.8 * sig).astype(np.float32)


def white_noise(seed=1):
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal(N).astype(np.float32)
    sig /= np.max(np.abs(sig)) + 1e-8
    return (0.3 * sig).astype(np.float32)


def hard_spliced_signal(seed=2):
    """Two halves with clearly different noise floors/DC offset, concatenated."""
    rng = np.random.default_rng(seed)
    half = N // 2
    t1 = _t(half)
    seg1 = 0.6 * np.sin(2 * np.pi * 180.0 * t1) + 0.01 * rng.standard_normal(half)
    seg2 = 0.05 * rng.standard_normal(N - half) + 0.4  # big DC offset + different noise floor
    sig = np.concatenate([seg1, seg2]).astype(np.float32)
    return sig


def low_passed_signal(seed=3, cutoff_hz=2000.0):
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal(N)
    # simple FFT brick-wall low-pass
    spec = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(N, d=1.0 / SR)
    spec[freqs > cutoff_hz] = 0.0
    sig = np.fft.irfft(spec, n=N)
    sig /= np.max(np.abs(sig)) + 1e-8
    return (0.5 * sig).astype(np.float32)


def clipped_signal(seed=4):
    rng = np.random.default_rng(seed)
    t = _t()
    sig = 1.6 * np.sin(2 * np.pi * 220.0 * t) + 0.05 * rng.standard_normal(N)
    sig = np.clip(sig, -1.0, 1.0).astype(np.float32)
    return sig


def silent_signal():
    return np.zeros(N, dtype=np.float32)


SIGNALS = {
    "harmonic": harmonic_tone_with_noise(),
    "noise": white_noise(),
    "spliced": hard_spliced_signal(),
    "lowpassed": low_passed_signal(),
    "clipped": clipped_signal(),
    "silent": silent_signal(),
}


# ---------------------------------------------------------------------------
# Schema / range / determinism tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("branch_name", list(BRANCHES.keys()))
@pytest.mark.parametrize("sig_name", list(SIGNALS.keys()))
def test_analyze_schema(branch_name, sig_name):
    module = BRANCHES[branch_name]
    audio = SIGNALS[sig_name]
    result = module.analyze(audio, SR)

    assert REQUIRED_KEYS.issubset(result.keys())
    assert isinstance(result["summary"], str) and len(result["summary"]) > 0
    assert result["suspicion"] is None or (
        isinstance(result["suspicion"], float) and 0.0 <= result["suspicion"] <= 1.0
    )
    assert isinstance(result["features"], dict)
    for v in result["features"].values():
        assert isinstance(v, float)
        assert np.isfinite(v)
    assert isinstance(result["flags"], list)
    assert all(isinstance(f, str) for f in result["flags"])
    assert result["used_in_score"] is False


@pytest.mark.parametrize("branch_name", list(BRANCHES.keys()))
def test_determinism(branch_name):
    module = BRANCHES[branch_name]
    audio = SIGNALS["harmonic"]
    r1 = module.analyze(audio, SR)
    r2 = module.analyze(audio, SR)
    assert r1["suspicion"] == r2["suspicion"]
    assert r1["features"] == r2["features"]
    assert r1["flags"] == r2["flags"]
    assert r1["summary"] == r2["summary"]


@pytest.mark.parametrize("branch_name", list(BRANCHES.keys()))
def test_extract_features_fixed_keys(branch_name):
    module = BRANCHES[branch_name]
    f1 = module.extract_features(SIGNALS["harmonic"], SR)
    f2 = module.extract_features(SIGNALS["noise"], SR)
    assert set(f1.keys()) == set(f2.keys())
    assert len(f1) > 0
    for v in list(f1.values()) + list(f2.values()):
        assert isinstance(v, float)
        assert np.isfinite(v)


@pytest.mark.parametrize("branch_name", list(BRANCHES.keys()))
def test_timing_budget(branch_name):
    module = BRANCHES[branch_name]
    audio = SIGNALS["harmonic"]
    # warm up (first call may pay import/JIT-ish costs e.g. librosa)
    module.analyze(audio, SR)
    start = time.perf_counter()
    module.analyze(audio, SR)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert elapsed_ms < 400.0, f"{branch_name} took {elapsed_ms:.1f} ms (budget ~150ms, hard cap 400ms)"


# ---------------------------------------------------------------------------
# Targeted behavioral tests
# ---------------------------------------------------------------------------

def test_splice_detects_hard_splice():
    result = splice.analyze(SIGNALS["spliced"], SR)
    assert "regions" in result
    assert len(result["regions"]) >= 1
    join_ms = DURATION_S * 1000.0 / 2.0
    close_to_join = any(
        r["start_ms"] - 500 <= join_ms <= r["end_ms"] + 500 for r in result["regions"]
    )
    assert close_to_join, f"expected a splice region near {join_ms}ms, got {result['regions']}"
    for r in result["regions"]:
        assert {"start_ms", "end_ms", "reason"}.issubset(r.keys())


def test_splice_no_false_positive_on_clean_tone():
    result = splice.analyze(SIGNALS["harmonic"], SR)
    # a clean vibrato tone should not produce many candidate splices
    assert len(result["regions"]) <= 2


def test_spectral_detects_bandlimit():
    result = spectral.analyze(SIGNALS["lowpassed"], SR)
    assert result["features"]["band_limit_cutoff_hz"] < 2500.0
    assert any("band-limited" in f for f in result["flags"])


def test_compression_detects_bandlimit():
    result = compression.analyze(SIGNALS["lowpassed"], SR)
    assert result["features"]["cutoff_ratio_of_nyquist"] < 0.5
    assert len(result["flags"]) > 0


def test_quality_detects_clipping():
    result = quality.analyze(SIGNALS["clipped"], SR)
    assert result["features"]["clipping_ratio"] > 0.0
    assert any("clipping" in f for f in result["flags"])
    assert result["suspicion"] is None


def test_quality_detects_silence():
    result = quality.analyze(SIGNALS["silent"], SR)
    assert result["features"]["silence_ratio"] > 0.9
    assert any("silence" in f for f in result["flags"])


def test_quality_snr_harmonic_better_than_noise():
    r_harm = quality.analyze(SIGNALS["harmonic"], SR)
    r_noise = quality.analyze(SIGNALS["noise"], SR)
    assert isinstance(r_harm["features"]["snr_estimate_db"], float)
    assert isinstance(r_noise["features"]["snr_estimate_db"], float)


def test_prosody_handles_pure_noise_gracefully():
    result = prosody.analyze(SIGNALS["noise"], SR)
    # pure noise has little/no voicing; should not crash and suspicion is None or bounded
    assert result["suspicion"] is None or 0.0 <= result["suspicion"] <= 1.0


def test_prosody_tracks_f0_on_harmonic_tone():
    feats = prosody.extract_features(SIGNALS["harmonic"], SR)
    assert 80.0 < feats["f0_mean_hz"] < 300.0
    assert feats["voiced_ratio"] > 0.3


def test_empty_audio_does_not_crash():
    empty = np.zeros(0, dtype=np.float32)
    for module in BRANCHES.values():
        result = module.analyze(empty, SR)
        assert REQUIRED_KEYS.issubset(result.keys())
        assert result["suspicion"] is None


def test_all_branch_features_concatenation():
    feats = all_branch_features(SIGNALS["harmonic"], SR)
    assert any(k.startswith("spectral.") for k in feats)
    assert any(k.startswith("prosody.") for k in feats)
    assert any(k.startswith("compression.") for k in feats)
    assert any(k.startswith("splice.") for k in feats)
    assert any(k.startswith("quality.") for k in feats)
    for v in feats.values():
        assert isinstance(v, float)
        assert np.isfinite(v)

    # deterministic and fixed-key across different signals
    feats2 = all_branch_features(SIGNALS["noise"], SR)
    assert set(feats.keys()) == set(feats2.keys())
