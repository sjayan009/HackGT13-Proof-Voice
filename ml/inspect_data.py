#!/usr/bin/env python3
"""Inventory the organizer training corpora (DiffSSD + LJRealResampled) and the held-out test set.

Writes outputs/dataset_audit.json and outputs/inventory.csv (one row per training file).
Held-out test files are only inspected for format (sr/channels/duration) - never for labels.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DIFFSSD = ROOT / "data/hearsay/train/DiffSSD/generated_speech"
LJREAL = ROOT / "data/hearsay/train/ljreal"
TEST = ROOT / "data/hearsay/test/extracted/HackGTHearsayTesting"
LIBRI = ROOT / "data/external/LibriSpeech"
OUT = ROOT / "outputs"
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def info(p: Path) -> dict:
    try:
        i = sf.info(str(p))
        return {"sr": i.samplerate, "channels": i.channels, "duration": i.frames / i.samplerate,
                "subtype": i.subtype, "format": i.format, "ok": True}
    except Exception as e:  # noqa: BLE001
        return {"sr": None, "channels": None, "duration": None, "subtype": None, "format": None,
                "ok": False, "err": str(e)[:120]}


def list_audio(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXT
                  and ".ipynb_checkpoints" not in p.parts)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = []
    for p in list_audio(LJREAL):
        rows.append({"path": p.relative_to(ROOT).as_posix(), "label": "bonafide", "source": "ljreal",
                     "generator": "bonafide_lj", "speaker": "LJ", "sentence": p.stem})
    # External bona fide (public LibriSpeech dev-clean/test-clean, CC BY 4.0) - see PROGRESS.md for rationale.
    for p in list_audio(LIBRI) if LIBRI.exists() else []:
        spk, chap = p.parts[-3], p.parts[-2]
        rows.append({"path": p.relative_to(ROOT).as_posix(), "label": "bonafide", "source": "librispeech",
                     "generator": f"bonafide_libri_{p.parts[-4]}", "speaker": f"libri_{spk}", "sentence": p.stem})
    for gdir in sorted(d for d in DIFFSSD.iterdir() if d.is_dir()):
        for p in list_audio(gdir):
            rel = p.relative_to(gdir).parts
            spk = rel[0] if len(rel) > 1 else "single"
            m = re.match(r"sentence_(\d+)", p.stem)
            rows.append({"path": p.relative_to(ROOT).as_posix(), "label": "spoof", "source": "diffssd",
                         "generator": gdir.name, "speaker": spk, "sentence": m.group(1) if m else p.stem})
    with ThreadPoolExecutor(16) as ex:
        infos = list(ex.map(lambda r: info(ROOT / r["path"]), rows))
    for r, i in zip(rows, infos):
        r.update(i)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "inventory.csv", index=False)

    test_files = sorted(TEST.glob("*.wav"))
    with ThreadPoolExecutor(16) as ex:
        tinfo = pd.DataFrame(list(ex.map(info, test_files)))

    def summ(d: pd.DataFrame) -> dict:
        dur = d["duration"].dropna()
        return {"n": int(len(d)), "unreadable": int((~d["ok"]).sum()),
                "sample_rates": {str(k): int(v) for k, v in Counter(d["sr"]).items()},
                "channels": {str(k): int(v) for k, v in Counter(d["channels"]).items()},
                "subtypes": {str(k): int(v) for k, v in Counter(d["subtype"]).items()},
                "duration_s": ({q: round(float(dur.quantile(x)), 3) for q, x in
                                [("min", 0), ("p10", .1), ("median", .5), ("p90", .9), ("max", 1)]}
                               | {"mean": round(float(dur.mean()), 3), "total_h": round(float(dur.sum()) / 3600, 2)})
                              if len(dur) else {},
                "n_speakers": int(d["speaker"].nunique()) if "speaker" in d else None}

    audit = {
        "training": {g: summ(d) for g, d in df.groupby("generator")},
        "totals": {"bonafide": int((df.label == "bonafide").sum()), "spoof": int((df.label == "spoof").sum())},
        "heldout_test": summ(tinfo.assign(speaker="?")) | {"note": "format only; no labels inspected"},
    }
    (OUT / "dataset_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    sys.exit(main())
