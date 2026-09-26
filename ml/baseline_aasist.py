#!/usr/bin/env python3
"""Baseline 0: organizer-provided ASVspoof5 AASIST checkpoint, zero-shot on our validation manifest.

AASIST class index 1 = bona fide (ASVspoof convention), so p_synth = softmax(logits)[:, 0].
Input: 64600 samples (~4.04 s); shorter clips are tiled (organizer `pad`), longer clips use the first 4 s.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
from audio_cache import Cache  # noqa: E402
from common import manifest, save_json, subset_report, testlike_crop  # noqa: E402

AASIST_DIR = ROOT / "data/hearsay/scoring/HackGTMinDCF/HackGTMinDCF/asvspoof5/Baseline-AASIST"
NB = 64600


def load_aasist(device):
    sys.path.insert(0, str(AASIST_DIR))
    from models.AASIST import Model  # type: ignore

    conf = json.loads((AASIST_DIR / "config/AASIST_ASVspoof5.conf").read_text())
    m = Model(conf["model_config"]).to(device)
    m.load_state_dict(torch.load(AASIST_DIR / "models/weights/AASIST/best.pth", map_location=device))
    return m.eval()


def pad(x):
    if len(x) >= NB:
        return x[:NB]
    return np.tile(x, int(NB / len(x)) + 1)[:NB]


@torch.no_grad()
def score(model, clips, device, bs=16):
    out = []
    for i in range(0, len(clips), bs):
        xb = torch.from_numpy(np.stack([pad(c) for c in clips[i:i + bs]])).to(device)
        _, logits = model(xb)
        out.append(torch.softmax(logits.float(), -1)[:, 0].cpu().numpy())
    return np.concatenate(out)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_aasist(dev)
    m = manifest("val")
    c = Cache("val")
    assert c.paths == m.path.tolist()
    clips = [testlike_crop(c.get(i), i) for i in range(len(c))]
    t = time.time()
    p = score(model, clips, dev)
    lat = (time.time() - t) / len(clips) * 1000
    rep = subset_report(m, p) | {"model": "AASIST (ASVspoof5 organizer checkpoint, zero-shot)",
                                 "ms_per_clip_batched": round(lat, 2)}
    save_json(rep, ROOT / "outputs/results/baseline_aasist_zeroshot.json")
    np.save(ROOT / "outputs/results/val_p_aasist_zeroshot.npy", p)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
