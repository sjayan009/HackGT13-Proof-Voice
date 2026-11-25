"""Render a frozen call (calls.json entry) to 16 kHz audio and pass it through a simulated channel.

Every channel returns (audio_16k float32, lost_packets bool[n_20ms_packets]) and is deterministic given the
call seed. The telephone path reuses the live system's code: soxr HQ 16k->8k, api/app/telephony/mulaw.py
encode/decode (bit-exact G.711), soxr HQ 8k->16k. These are SIMULATIONS, not real telephone recordings.

  clean              rendered call (clip level -26 dBFS, -70 dBFS white floor), wideband 16 kHz
  telephone          16k -> 8k -> mu-law -> 8k -> 16k
  telephone_noise30  pink noise at 30 dB below the -26 dBFS clip level, added BEFORE the phone channel
  telephone_noise20  same at 20 dB
  telephone_loss5    telephone + 5 % of 20 ms packets zero-filled, bursty (Gilbert-Elliott, mean burst 3 packets)
  opus_telephone     Opus 24 kbps round trip (ffmpeg/libopus, as ml/robustness.py), then telephone
  replay_sim         CRUDE replay proxy: random exponential-decay room IR (RT60 0.25-0.6 s) + 150-6000 Hz
                     band-pass, wideband. Not a real loudspeaker/microphone replay.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "api"))
from app.telephony import mulaw  # noqa: E402

SR = 16000
PACKET = 320  # 20 ms at 16 kHz (= 160 samples at 8 kHz)
CHANNELS = ["clean", "telephone", "telephone_noise30", "telephone_noise20", "telephone_loss5",
            "opus_telephone", "replay_sim"]
FAMILY = {c: ("telephone" if c.startswith(("telephone", "opus")) else "clean") for c in CHANNELS}


def render_call(call: dict, cache, level_dbfs: float = -26.0, floor_dbfs: float = -70.0) -> np.ndarray:
    rng = np.random.default_rng(call["seed"])
    n = int(call["duration_ms"] * SR / 1000)
    x = rng.standard_normal(n).astype(np.float32) * 10 ** (floor_dbfs / 20)
    target = 10 ** (level_dbfs / 20)
    for s in call["segments"]:
        c = cache.get(s["val_idx"])[: int(s["dur_ms"] * SR / 1000)]
        g = target / (np.sqrt(np.mean(c ** 2)) + 1e-9)
        g = min(g, 0.99 / (np.abs(c).max() + 1e-9))  # never clip
        o = int(s["start_ms"] * SR / 1000)
        c = c[: n - o]
        x[o:o + len(c)] += g * c
    return np.clip(x, -1, 1)


def _phone(x: np.ndarray, lost: np.ndarray | None = None) -> np.ndarray:
    import soxr

    x8 = soxr.resample(x, SR, 8000, quality="HQ").astype(np.float32)
    y8 = mulaw.decode(mulaw.encode(np.clip(x8, -1, 1))).copy()
    if lost is not None:
        for k in np.flatnonzero(lost):
            y8[k * 160:(k + 1) * 160] = 0.0
    y = soxr.resample(y8, 8000, SR, quality="HQ").astype(np.float32)
    return np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)]


def gilbert_loss(n_packets: int, rng, rate: float = 0.05, mean_burst: float = 3.0) -> np.ndarray:
    r = 1.0 / mean_burst  # bad -> good
    p = rate * r / (1 - rate)  # good -> bad, so the stationary loss rate is `rate`
    lost = np.zeros(n_packets, bool)
    bad = rng.random() < rate
    u = rng.random(n_packets)
    for k in range(n_packets):
        lost[k] = bad
        bad = (u[k] >= r) if bad else (u[k] < p)
    return lost


def pink(n: int, rng) -> np.ndarray:
    f = np.fft.rfftfreq(n, 1 / SR)
    spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / np.sqrt(np.maximum(f, 20.0))
    z = np.fft.irfft(spec, n).astype(np.float32)
    return z / (np.sqrt(np.mean(z ** 2)) + 1e-12)


def replay(x: np.ndarray, rng) -> np.ndarray:
    from scipy.signal import butter, fftconvolve, sosfilt

    rt60 = rng.uniform(0.25, 0.6)
    t = np.arange(int(rt60 * SR)) / SR
    h = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60) * 0.1
    h[0] = 1.0  # direct path
    y = fftconvolve(x, h)[: len(x)]
    y = sosfilt(butter(4, [150, 6000], "band", fs=SR, output="sos"), y)
    return (y * np.sqrt(np.mean(x ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-12)).astype(np.float32)


def apply(channel: str, x: np.ndarray, seed: int, level_dbfs: float = -26.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng([seed, CHANNELS.index(channel)])
    n_packets = int(np.ceil(len(x) / PACKET))
    lost = np.zeros(n_packets, bool)
    if channel == "clean":
        y = x
    elif channel == "telephone":
        y = _phone(x)
    elif channel.startswith("telephone_noise"):
        snr = float(channel.removeprefix("telephone_noise"))
        y = _phone(x + pink(len(x), rng) * 10 ** ((level_dbfs - snr) / 20))
    elif channel == "telephone_loss5":
        lost = gilbert_loss(n_packets, rng)
        y = _phone(x, lost)
    elif channel == "opus_telephone":
        from robustness import OPUS, codec_roundtrip

        y = _phone(codec_roundtrip(x, [OPUS]))
    elif channel == "replay_sim":
        y = replay(x, rng)
    else:
        raise KeyError(channel)
    return np.clip(y, -1, 1).astype(np.float32), lost


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    lost = gilbert_loss(200_000, rng)
    assert abs(lost.mean() - 0.05) < 0.005, lost.mean()
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, lost.astype(int), 0])))[::2]
    assert 2.5 < runs.mean() < 3.5, runs.mean()
    t = np.arange(4 * SR) / SR
    x = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    for ch in CHANNELS:
        y, lp = apply(ch, x, 1)
        y2, _ = apply(ch, x, 1)
        assert len(y) == len(x) and np.array_equal(y, y2), ch
        if ch == "telephone":
            assert np.corrcoef(x[1000:-1000], y[1000:-1000])[0, 1] > 0.99
    print("channels ok; burst mean", round(runs.mean(), 2))
