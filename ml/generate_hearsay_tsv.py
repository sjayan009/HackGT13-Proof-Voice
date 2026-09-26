#!/usr/bin/env python3
"""Score the HEARSAY held-out set with the frozen selected model and write the prediction TSV.

    python ml/generate_hearsay_tsv.py \
        --input data/hearsay/test \
        --template data/hearsay/template/HearsayScoreKey4TeamX.tsv \
        --output outputs/team_predictions.tsv

`--input` may be a directory of WAVs (searched recursively) or the organizer .zip. Rows follow the template order;
filenames are copied verbatim from the template. `cm-score` = calibrated p_synthetic in [0,1] (1.0 = synthetic).
Also writes `<output stem>_bonafide_high.tsv` (= 1 - p) because the organizer scorer treats higher = bona fide
(see PROGRESS.md "score orientation"), plus a JSON sidecar with model hash / config for reproducibility.
Offline: no network, no xAI. Never writes to the template.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "ml"))


def find_wavs(inp: Path, tmp: Path) -> dict[str, Path]:
    if inp.is_file() and inp.suffix == ".zip":
        with zipfile.ZipFile(inp) as z:
            z.extractall(tmp)
        for t in list(tmp.rglob("*.tar")):
            import tarfile

            with tarfile.open(t) as tf:
                tf.extractall(tmp)
        inp = tmp
    found: dict[str, Path] = {}
    for p in sorted(inp.rglob("*")):
        if p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg", ".m4a"} and p.name not in found:
            found[p.name] = p
    return found


def read_template(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8-sig").replace("\r", "").strip("\n").split("\n")
    if lines[0].split("\t") != ["filename", "cm-score"]:
        raise SystemExit(f"unexpected template header: {lines[0]!r}")
    return [l.split("\t")[0] for l in lines[1:]]


def sha1_file(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--template", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--model-dir", type=Path, default=ROOT / "models/selected")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args(argv)

    if a.output.resolve() == a.template.resolve():
        raise SystemExit("refusing to overwrite the organizer template")
    import torch
    from app.audio.ingest import load_audio
    from app.detectors.primary import PrimaryDetector

    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    names = read_template(a.template)
    with tempfile.TemporaryDirectory() as td:
        wavs = find_wavs(a.input, Path(td))
        missing = [n for n in names if n not in wavs]
        if missing:
            raise SystemExit(f"{len(missing)} template files not found under {a.input}, e.g. {missing[:3]}")
        det = PrimaryDetector(a.model_dir, dev)
        t = time.time()
        audio = [load_audio(wavs[n])[0] for n in names]
        scores = np.zeros(len(names))
        for b in range(0, len(names), 256):
            scores[b:b + 256] = det.calibrate(det.logits(audio[b:b + 256], bs=a.bs))
    scores = np.clip(np.nan_to_num(scores, nan=0.5), 0.0, 1.0)
    assert len(scores) == len(names) == len(set(names))
    a.output.parent.mkdir(parents=True, exist_ok=True)

    def write(path: Path, vals):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("filename\tcm-score\n")
            for n, v in zip(names, vals):
                f.write(f"{n}\t{v:.6f}\n")

    write(a.output, scores)
    alt = a.output.with_name(a.output.stem + "_bonafide_high.tsv")
    write(alt, 1.0 - scores)
    side = {"output": str(a.output), "bonafide_high_variant": str(alt), "n": len(names),
            "model_dir": str(a.model_dir), "model_sha1": sha1_file(a.model_dir / "model.pt"),
            "meta": json.loads((a.model_dir / "meta.json").read_text()),
            "calibration": det.cal, "device": dev, "seconds": round(time.time() - t, 1),
            "score_stats": {"mean": float(scores.mean()), "median": float(np.median(scores)),
                            "frac_above_threshold": float(np.mean(scores >= det.threshold))},
            "orientation": "cm-score = p_synthetic (1.0 = synthetic); *_bonafide_high.tsv = 1 - p"}
    a.output.with_suffix(".json").write_text(json.dumps(side, indent=2, default=str))
    from validate_hearsay_tsv import validate

    errs = validate(a.output, a.template) + validate(alt, a.template)
    if errs:
        print("\n".join(errs), file=sys.stderr)
        return 1
    print(f"wrote {a.output} and {alt} ({len(names)} rows, {side['seconds']} s); validator OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
