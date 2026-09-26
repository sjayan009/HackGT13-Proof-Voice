#!/usr/bin/env python3
"""Fine-tune an SSL anti-spoofing detector on the frozen manifests, selecting checkpoints by validation minDCF.

    python ml/train.py --backbone facebook/wav2vec2-xls-r-300m --out models/xlsr_v1 --steps 4000
    python ml/train.py ... --exclude-generators elevenlabs unit_speech   # unseen-generator experiment

Augmentations are applied to BOTH classes (so none of them can become a class shortcut):
gain, additive noise, random low-pass (codec/band-limit proxy), resample round-trip, telephone band-pass.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "api"))
from audio_cache import Cache  # noqa: E402
from common import TESTLIKE_MAX_S, TESTLIKE_MIN_S, manifest, save_json, subset_report, testlike_crop  # noqa: E402
from evaluate_mindcf import fast_mindcf  # noqa: E402
from app.detectors.ssl_model import SSLDetector, normalize, save_detector  # noqa: E402

SR = 16000


class Augment:
    def __init__(self, rng: np.random.Generator, strength: float = 1.0):
        self.rng, self.s = rng, strength

    def __call__(self, x: np.ndarray) -> np.ndarray:
        r = self.rng
        if r.random() < 0.3 * self.s:  # resample round-trip (different resampler artefacts)
            import librosa
            mid = int(r.choice([11025, 12000, 22050, 24000, 32000]))
            x = librosa.resample(librosa.resample(x, orig_sr=SR, target_sr=mid, res_type="polyphase"),
                                 orig_sr=mid, target_sr=SR, res_type="polyphase")
        if r.random() < 0.25 * self.s:  # low-pass: codec / band-limit proxy
            fc = r.uniform(3400, 7600)
            x = sosfilt(butter(8, fc, "low", fs=SR, output="sos"), x)
        elif r.random() < 0.07 * self.s:  # telephone band
            x = sosfilt(butter(4, [300, 3400], "band", fs=SR, output="sos"), x)
        if r.random() < 0.35 * self.s:  # additive noise (white or pink-ish)
            snr = r.uniform(12, 40)
            n = r.standard_normal(len(x))
            if r.random() < 0.5:
                n = np.cumsum(n)
                n -= np.convolve(n, np.ones(64) / 64, mode="same")
            p = np.mean(x ** 2) + 1e-10
            x = x + n * math.sqrt(p / (10 ** (snr / 10)) / (np.mean(n ** 2) + 1e-10))
        if r.random() < 0.5:
            x = x * 10 ** (r.uniform(-8, 6) / 20)
        return np.clip(x, -1, 1).astype(np.float32)


def batchify(clips, device):
    lens = torch.tensor([len(c) for c in clips])
    xb = torch.zeros(len(clips), int(lens.max()))
    for i, c in enumerate(clips):
        xb[i, :len(c)] = torch.from_numpy(c)
    xb, lens = xb.to(device), lens.to(device)
    return normalize(xb, lens), lens


@torch.no_grad()
def predict(model, clips, device, bs=8):
    model.eval()
    torch.cuda.empty_cache()
    order = np.argsort([len(c) for c in clips])
    out = np.zeros(len(clips), np.float32)
    for b in range(0, len(order), bs):
        ids = order[b:b + bs]
        xb, lens = batchify([clips[i] for i in ids], device)
        with torch.autocast("cuda", dtype=torch.float16):
            out[ids] = model(xb, lens).float().cpu().numpy()
    model.train()
    torch.cuda.empty_cache()
    return out  # logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="facebook/wav2vec2-xls-r-300m")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--aug", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-generators", nargs="*", default=[])
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--num-layers", type=int, default=None, help="keep only the first N transformer layers")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)

    mtr, mva = manifest("train"), manifest("val")
    ctr, cva = Cache("train"), Cache("val")
    keep = ~mtr.generator.isin(a.exclude_generators).to_numpy()
    tr_idx = np.flatnonzero(keep)
    ytr = mtr.y.to_numpy()
    bona_idx = tr_idx[ytr[tr_idx] == 0]
    spoof_by_gen = {g: tr_idx[(mtr.generator.to_numpy()[tr_idx] == g)] for g in mtr[mtr.y == 1].generator.unique()
                    if g not in a.exclude_generators}
    gens = sorted(spoof_by_gen)
    val_clips = [testlike_crop(cva.get(i), i, 0) for i in range(len(cva))]
    yva = mva.y.to_numpy()

    model = SSLDetector(a.backbone, cache_dir=str(ROOT / "models/hf_cache"), num_layers=a.num_layers).to(dev)
    model.ssl.feature_extractor._freeze_parameters() if hasattr(model.ssl, "feature_extractor") else None
    if a.grad_ckpt:
        model.ssl.gradient_checkpointing_enable()
    ssl_params = [p for n, p in model.named_parameters() if n.startswith("ssl.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("ssl.")]
    opt = torch.optim.AdamW([{"params": ssl_params, "lr": a.lr}, {"params": head_params, "lr": a.head_lr}],
                            weight_decay=1e-4)
    warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    scaler = torch.amp.GradScaler()
    aug = Augment(rng, a.aug)
    lossf = torch.nn.BCEWithLogitsLoss()

    def sample_batch():
        clips, ys = [], []
        for _ in range(a.bs):
            if rng.random() < 0.5:
                i, y = int(rng.choice(bona_idx)), 0
            else:
                i, y = int(rng.choice(spoof_by_gen[gens[rng.integers(len(gens))]])), 1
            x = ctr.get(i)
            n = int(rng.uniform(TESTLIKE_MIN_S, TESTLIKE_MAX_S) * SR)
            if len(x) > n:
                o = int(rng.integers(0, len(x) - n))
                x = x[o:o + n]
            clips.append(aug(x))
            ys.append(y)
        return clips, torch.tensor(ys, dtype=torch.float32, device=dev)

    log, best = [], (9.0, -1)
    t0 = time.time()
    model.train()
    for step in range(1, a.steps + 1):
        for _ in range(a.accum):
            clips, yb = sample_batch()
            xb, lens = batchify(clips, dev)
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(xb, lens)
            loss = lossf(logit.float(), yb) / a.accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if step % 50 == 0:
            print(f"step {step} loss {loss.item() * a.accum:.4f} {time.time() - t0:.0f}s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            lg = predict(model, val_clips, dev)
            p = 1 / (1 + np.exp(-lg))
            md = fast_mindcf(yva, p)[0]
            log.append({"step": step, "val_minDCF": round(md, 4), "elapsed_s": round(time.time() - t0)})
            print("EVAL", log[-1], flush=True)
            if md < best[0]:
                best = (md, step)
                save_detector(model, out_dir, {"step": step, "val_minDCF": md, "args": vars(a)})
                np.save(out_dir / "val_logits.npy", lg)
            (out_dir / "train_log.json").write_text(json.dumps(log, indent=1))
    lg = np.load(out_dir / "val_logits.npy")
    rep = subset_report(mva, 1 / (1 + np.exp(-lg))) | {"best_step": best[1], "log": log, "args": vars(a)}
    save_json(rep, out_dir / "val_report.json")
    print(json.dumps({"best": best, "pooled": rep["pooled"], "per_gen": rep["per_generator_minDCF"]}, indent=1))


if __name__ == "__main__":
    main()
