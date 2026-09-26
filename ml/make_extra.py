#!/usr/bin/env python3
"""Extra TRAIN-ONLY bona fide: LibriSpeech dev-other + test-other (noisier / harder speakers, CC BY 4.0).
Validation manifests stay frozen; speakers here are disjoint from dev-clean/test-clean by LibriSpeech design.

    python ml/make_extra.py && python ml/audio_cache.py --split train_extra
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rows = []
for sub in ["dev-other", "test-other"]:
    for p in sorted((ROOT / "data/external/LibriSpeech" / sub).rglob("*.flac")):
        rows.append({"path": p.relative_to(ROOT).as_posix(), "label": "bonafide", "y": 0, "source": "librispeech",
                     "generator": f"bonafide_libri_{sub}", "speaker": f"libri_{p.parts[-3]}", "sentence": p.stem,
                     "group": f"libri_{p.parts[-3]}", "sr": 16000, "duration": None})
pd.DataFrame(rows).to_csv(ROOT / "outputs/manifests/train_extra.csv", index=False)
print(len(rows), "extra bona fide train clips")
