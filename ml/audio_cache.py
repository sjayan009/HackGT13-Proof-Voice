#!/usr/bin/env python3
"""Decode every manifest file once to 16 kHz mono int16 and store in a flat memmap + offsets.

The same decoder/resampler (soundfile -> soxr_hq via librosa) is used for bona fide and spoof so the
model cannot learn a "which resampler" shortcut. Clips are capped at MAX_S seconds.

    python ml/audio_cache.py --split train val        # -> outputs/cache/{split}.pcm16 + {split}_index.npy
    python ml/audio_cache.py --dir data/hearsay/test/extracted/HackGTHearsayTesting --name heldout
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/cache"
SR = 16000
MAX_S = 14.0  # >= longest held-out clip (13.6 s), so held-out audio is never truncated


def load_16k(path: str | Path, max_s: float | None = MAX_S) -> np.ndarray:
    import librosa
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    if max_s:
        x = x[: int(max_s * SR)]
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def _work(p: str) -> np.ndarray:
    return (load_16k(ROOT / p) * 32767.0).round().astype(np.int16)


def build(paths: list[str], name: str) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    lens = []
    with ThreadPoolExecutor(12) as ex, open(CACHE / f"{name}.pcm16", "wb") as f:
        for a in ex.map(_work, paths):  # ordered; streamed to disk to bound RAM
            f.write(a.tobytes())
            lens.append(len(a))
    lens = np.array(lens, dtype=np.int64)
    offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
    np.save(CACHE / f"{name}_index.npy", np.stack([offs, lens], 1))
    pd.Series(paths).to_csv(CACHE / f"{name}_paths.csv", index=False, header=["path"])
    print(name, len(paths), "clips", round(lens.sum() / SR / 3600, 2), "h")


class Cache:
    """Random access to cached clips as float32."""

    def __init__(self, name: str):
        self.idx = np.load(CACHE / f"{name}_index.npy")
        self.mm = np.memmap(CACHE / f"{name}.pcm16", dtype=np.int16, mode="r")
        self.paths = pd.read_csv(CACHE / f"{name}_paths.csv").path.tolist()

    def __len__(self):
        return len(self.idx)

    def get(self, i: int) -> np.ndarray:
        o, n = self.idx[i]
        return self.mm[o:o + n].astype(np.float32) / 32767.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="*", default=[])
    ap.add_argument("--dir")
    ap.add_argument("--name")
    a = ap.parse_args()
    for s in a.split:
        m = pd.read_csv(ROOT / f"outputs/manifests/{s}.csv")
        build(m.path.tolist(), s)
    if a.dir:
        files = sorted(Path(ROOT / a.dir).glob("*.wav"))
        build([f.relative_to(ROOT).as_posix() for f in files], a.name)


if __name__ == "__main__":
    main()
