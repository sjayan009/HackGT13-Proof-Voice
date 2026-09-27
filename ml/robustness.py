#!/usr/bin/env python3
"""Robustness / laundering evaluation: re-score a fixed stratified validation subset under transforms.

Transforms (applied identically to bona fide and spoof): MP3 64k, Opus 24k, AAC 48k, MP3->Opus->MP3 chain,
G.711 mu-law telephone (8 kHz), white noise 20 dB / 10 dB SNR, -18 dB gain, clipping.

    python ml/robustness.py --model-dir models/selected [--n 1400]
Writes outputs/results/robustness.json (minDCF, EER, FRR/FAR at the clean threshold, mean score drift).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "api"))
from audio_cache import Cache  # noqa: E402
from common import manifest, save_json, testlike_crop  # noqa: E402
from evaluate_mindcf import fast_mindcf, full_report  # noqa: E402

SR = 16000


def ffmpeg():
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def codec_roundtrip(x: np.ndarray, steps: list[tuple[str, list[str]]]) -> np.ndarray:
    """steps: [(container_ext, encoder args)], decoded back to 16 kHz mono float32 after each step."""
    import soundfile as sf

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.wav"
        sf.write(src, x, SR, subtype="PCM_16")
        for k, (ext, args) in enumerate(steps):
            enc = Path(td) / f"s{k}.{ext}"
            subprocess.run([ffmpeg(), "-v", "error", "-y", "-i", str(src), *args, str(enc)], check=True)
            dec = Path(td) / f"d{k}.wav"
            subprocess.run([ffmpeg(), "-v", "error", "-y", "-i", str(enc), "-ac", "1", "-ar", str(SR), str(dec)],
                           check=True)
            src = dec
        y, _ = sf.read(src, dtype="float32")
    n = len(x)
    return (np.pad(y, (0, max(0, n - len(y))))[:n]).astype(np.float32)


MP3 = ("mp3", ["-c:a", "libmp3lame", "-b:a", "64k"])
OPUS = ("ogg", ["-c:a", "libopus", "-b:a", "24k"])
AAC = ("m4a", ["-c:a", "aac", "-b:a", "48k"])
ULAW = ("wav", ["-ar", "8000", "-c:a", "pcm_mulaw"])


def noise(x, snr, rng):
    n = rng.standard_normal(len(x)).astype(np.float32)
    return x + n * np.sqrt(np.mean(x ** 2) / 10 ** (snr / 10) / np.mean(n ** 2))


TRANSFORMS = {
    "clean": lambda x, r: x,
    "mp3_64k": lambda x, r: codec_roundtrip(x, [MP3]),
    "opus_24k": lambda x, r: codec_roundtrip(x, [OPUS]),
    "aac_48k": lambda x, r: codec_roundtrip(x, [AAC]),
    "mp3_opus_mp3_chain": lambda x, r: codec_roundtrip(x, [MP3, OPUS, MP3]),
    "telephone_mulaw_8k": lambda x, r: codec_roundtrip(x, [ULAW]),
    "noise_20dB": lambda x, r: noise(x, 20, r),
    "noise_10dB": lambda x, r: noise(x, 10, r),
    "gain_-18dB": lambda x, r: x * 10 ** (-18 / 20),
    "clipping": lambda x, r: np.clip(x * 6, -0.5, 0.5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/selected")
    ap.add_argument("--n", type=int, default=1400)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", default="outputs/results/robustness.json")
    ap.add_argument("--bs", type=int, default=8, help="inference batch size; keep below WDDM shared-memory spill")
    a = ap.parse_args()
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from app.detectors.primary import PrimaryDetector

    if torch.cuda.is_available() and a.device != "cpu":
        torch.cuda.set_per_process_memory_fraction(0.9)
    det = PrimaryDetector(ROOT / a.model_dir, a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    m = manifest("val")
    c = Cache("val")
    rng = np.random.default_rng(3)
    bona = np.flatnonzero(m.y == 0)
    spoof = np.flatnonzero(m.y == 1)
    idx = np.sort(np.concatenate([rng.choice(bona, min(len(bona), a.n // 2), replace=False),
                                  rng.choice(spoof, a.n // 2, replace=False)]))
    y = m.y.to_numpy()[idx]
    clean = [testlike_crop(c.get(i), i, 0) for i in idx]
    out, thr_clean, p_clean = {}, None, None
    for name, fn in TRANSFORMS.items():
        if a.only and name not in a.only and name != "clean":
            continue
        with ThreadPoolExecutor(12) as ex:
            xs = list(ex.map(lambda t: np.clip(fn(t[1], np.random.default_rng(t[0])), -1, 1).astype(np.float32),
                             enumerate(clean)))
        p = det.calibrate(det.logits(xs, bs=a.bs))
        r = full_report(y, p)
        if name == "clean":
            thr_clean, p_clean = r["p_synth_threshold_at_minDCF"], p
        pred = p >= thr_clean
        out[name] = {"minDCF": round(r["minDCF"], 4), "EER": round(r["EER"], 4), "AUC": round(r["AUC"], 4),
                     "FRR_bonafide_at_clean_thr": round(float(pred[y == 0].mean()), 4),
                     "FAR_spoof_at_clean_thr": round(float((~pred[y == 1]).mean()), 4),
                     "mean_abs_score_drift": round(float(np.mean(np.abs(p - p_clean))), 4),
                     "bonafide_mean_p": round(float(p[y == 0].mean()), 4), "spoof_mean_p": round(float(p[y == 1].mean()), 4)}
        print(name, out[name], flush=True)
    save_json({"model_dir": a.model_dir, "n": int(len(idx)), "n_bonafide": int((y == 0).sum()),
               "conditions": out}, ROOT / a.output)


if __name__ == "__main__":
    main()
