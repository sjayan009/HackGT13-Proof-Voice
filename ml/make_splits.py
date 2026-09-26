#!/usr/bin/env python3
"""Deterministic train/validation manifests (frozen once written).

Grouping (prevents leakage of speaker / sentence identity across splits):
  * LJ bona fide           -> group = LJ chapter (LJ001..LJ050); val = chapter % 5 == 0
  * LibriSpeech bona fide  -> group = speaker; val = stable hash(speaker) % 5 == 0
  * single-speaker spoofs  -> group = sentence id; val = sentence % 5 == 0
  * multi-speaker spoofs   -> group = cloned speaker; val = 2 of 10 speakers (speaker-disjoint)

Spoofs are sub-sampled per generator (seeded) to keep embedding / fine-tuning cost bounded.
The `generator` column enables leave-one-generator-out (LOGO) evaluation on top of these manifests.

Outputs: outputs/manifests/{train,val}.csv  (+ manifest_meta.json with counts and hash)
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/manifests"
SEED = 1337
CAP_TRAIN, CAP_VAL = 2500, 600
VAL_SPEAKERS = {"speaker_6167", "speaker_2061"}  # 2 of the 10 cloned speakers held out


def h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def assign(r) -> tuple[str, str]:
    if r.generator == "bonafide_lj":
        ch = int(r.sentence[2:5])
        return f"lj_ch{ch}", "val" if ch % 5 == 0 else "train"
    if r.source == "librispeech":
        return r.speaker, "val" if h(r.speaker) % 5 == 0 else "train"
    if r.speaker == "single":
        sid = int(re.match(r"(\d+)", str(r.sentence)).group(1))
        return f"sent{sid}", "val" if sid % 5 == 0 else "train"
    return r.speaker, "val" if r.speaker in VAL_SPEAKERS else "train"


def main() -> None:
    df = pd.read_csv(ROOT / "outputs/inventory.csv", dtype={"sentence": str})
    df = df[df.ok].copy()
    df[["group", "split"]] = df.apply(lambda r: pd.Series(assign(r)), axis=1)
    df["y"] = (df.label == "spoof").astype(int)
    rng = np.random.default_rng(SEED)
    parts = []
    for (gen, split), d in df.groupby(["generator", "split"]):
        cap = CAP_TRAIN if split == "train" else CAP_VAL
        if d.y.iloc[0] == 1 and len(d) > cap:
            d = d.iloc[np.sort(rng.choice(len(d), cap, replace=False))]
        parts.append(d)
    m = pd.concat(parts).sort_values("path").reset_index(drop=True)
    cols = ["path", "label", "y", "source", "generator", "speaker", "sentence", "group", "sr", "duration"]
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"seed": SEED, "cap_train": CAP_TRAIN, "cap_val": CAP_VAL, "val_speakers": sorted(VAL_SPEAKERS)}
    for split in ["train", "val"]:
        s = m[m.split == split][cols]
        s.to_csv(OUT / f"{split}.csv", index=False)
        meta[split] = {"n": int(len(s)), "by_generator": s.generator.value_counts().sort_index().to_dict(),
                       "sha1": hashlib.sha1(s.path.str.cat().encode()).hexdigest()}
    assert not set(m[m.split == "train"].group) & set(m[m.split == "val"].group), "group leakage"
    (OUT / "manifest_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
