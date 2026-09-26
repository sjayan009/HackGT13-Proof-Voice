#!/usr/bin/env python3
"""Extract frozen SSL layer-wise mean-pooled embeddings for linear-probe benchmarking.

    python ml/extract_ssl.py --model facebook/wav2vec2-xls-r-300m --splits train val
Output: outputs/embeddings/<tag>_<split>.npy  float16 [N, L+1, D]
Train clips: one seeded random test-like crop; val: the same test-like crop used by every model.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
from audio_cache import Cache  # noqa: E402
from common import testlike_crop  # noqa: E402

EMB = ROOT / "outputs/embeddings"


def tag_of(name: str) -> str:
    return name.split("/")[-1]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    from transformers import AutoModel

    dev = torch.device("cuda")
    model = AutoModel.from_pretrained(a.model, cache_dir=str(ROOT / "models/hf_cache")).to(dev).half().eval()
    EMB.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        c = Cache(split)
        seed = 0 if split != "train" else 7
        clips = [testlike_crop(c.get(i), i, seed) for i in range(len(c))]
        order = np.argsort([len(x) for x in clips])  # length-bucketed batches, minimal padding
        out = None
        t = time.time()
        for b in range(0, len(order), a.bs):
            ids = order[b:b + a.bs]
            n = max(len(clips[i]) for i in ids)
            xb = np.zeros((len(ids), n), np.float32)
            mask = np.zeros((len(ids), n), np.int64)
            for j, i in enumerate(ids):
                x = clips[i]
                x = (x - x.mean()) / (x.std() + 1e-7)  # standard wav2vec2 input normalisation
                xb[j, :len(x)] = x
                mask[j, :len(x)] = 1
            hs = model(torch.from_numpy(xb).to(dev).half(), output_hidden_states=True).hidden_states
            hs = torch.stack(hs, 1).float()  # [B, L+1, T, D]
            T = hs.shape[2]
            fl = torch.tensor([min(T, int(np.ceil(len(clips[i]) / 320))) for i in ids], device=dev)
            fm = (torch.arange(T, device=dev)[None] < fl[:, None]).float()[:, None, :, None]
            pooled = (hs * fm).sum(2) / fm.sum(2)
            if out is None:
                out = np.zeros((len(clips),) + tuple(pooled.shape[1:]), np.float16)
            out[ids] = pooled.cpu().numpy().astype(np.float16)
            if b % (a.bs * 200) == 0:
                print(split, b, f"{time.time() - t:.0f}s", flush=True)
        np.save(EMB / f"{tag_of(a.model)}_{split}.npy", out)
        print(split, "done", out.shape, f"{(time.time() - t) / len(clips) * 1000:.1f} ms/clip", flush=True)


if __name__ == "__main__":
    main()
