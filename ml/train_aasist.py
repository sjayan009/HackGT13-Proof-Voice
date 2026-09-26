#!/usr/bin/env python3
"""Candidate family 2: fine-tune the organizer-provided ASVspoof5 AASIST checkpoint on our manifests
(same sampler, augmentation and validation protocol as ml/train.py).

    python ml/train_aasist.py --out models/aasist_ft --steps 3000
Writes <out>/model.pth, val_logits.npy (logit = log p_spoof - log p_bona), val_report.json
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
from audio_cache import Cache  # noqa: E402
from baseline_aasist import NB, load_aasist, pad  # noqa: E402
from common import TESTLIKE_MAX_S, TESTLIKE_MIN_S, manifest, save_json, subset_report, testlike_crop  # noqa: E402
from evaluate_mindcf import fast_mindcf  # noqa: E402
from train import Augment  # noqa: E402

SR = 16000


@torch.no_grad()
def predict(model, clips, dev, bs=16):
    model.eval()
    out = []
    for i in range(0, len(clips), bs):
        xb = torch.from_numpy(np.stack([pad(c) for c in clips[i:i + bs]])).to(dev)
        _, lg = model(xb)
        lg = torch.log_softmax(lg.float(), -1)
        out.append((lg[:, 0] - lg[:, 1]).cpu().numpy())
    model.train()
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = torch.device("cuda")
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    mtr, mva = manifest("train"), manifest("val")
    ctr, cva = Cache("train"), Cache("val")
    y = mtr.y.to_numpy()
    bona = np.flatnonzero(y == 0)
    gens = sorted(mtr[mtr.y == 1].generator.unique())
    by_gen = {g: np.flatnonzero((mtr.generator == g).to_numpy()) for g in gens}
    val_clips = [testlike_crop(cva.get(i), i, 0) for i in range(len(cva))]
    yva = mva.y.to_numpy()
    model = load_aasist(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
    aug = Augment(rng)
    ce = torch.nn.CrossEntropyLoss()
    log, best, t0 = [], 9.0, time.time()
    for step in range(1, a.steps + 1):
        xs, ys = [], []
        for _ in range(a.bs):
            if rng.random() < 0.5:
                i, lab = int(rng.choice(bona)), 1  # AASIST convention: class 1 = bona fide
            else:
                i, lab = int(rng.choice(by_gen[gens[rng.integers(len(gens))]])), 0
            x = ctr.get(i)
            n = int(rng.uniform(TESTLIKE_MIN_S, TESTLIKE_MAX_S) * SR)
            if len(x) > n:
                o = int(rng.integers(0, len(x) - n))
                x = x[o:o + n]
            xs.append(pad(aug(x)))
            ys.append(lab)
        xb = torch.from_numpy(np.stack(xs)).to(dev)
        _, lg = model(xb, Freq_aug=True)
        loss = ce(lg, torch.tensor(ys, device=dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            lgv = predict(model, val_clips, dev)
            md = fast_mindcf(yva, 1 / (1 + np.exp(-lgv)))[0]
            log.append({"step": step, "val_minDCF": round(md, 4)})
            print("EVAL", log[-1], flush=True)
            if md < best:
                best = md
                torch.save(model.state_dict(), out / "model.pth")
                np.save(out / "val_logits.npy", lgv)
    lgv = np.load(out / "val_logits.npy")
    rep = subset_report(mva, 1 / (1 + np.exp(-lgv))) | {"log": log, "model": "AASIST fine-tuned from ASVspoof5 ckpt",
                                                         "input_samples": NB}
    save_json(rep, out / "val_report.json")
    print(json.dumps({"pooled": rep["pooled"], "per_gen": rep["per_generator_minDCF"]}, indent=1))


if __name__ == "__main__":
    main()
