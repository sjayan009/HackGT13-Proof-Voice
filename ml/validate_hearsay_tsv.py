#!/usr/bin/env python3
"""Validate a HEARSAY prediction TSV against the organizer template. Fails loudly (exit 1) on any mismatch.

    python ml/validate_hearsay_tsv.py --pred outputs/team_predictions.tsv \
        --template data/hearsay/template/HearsayScoreKey4TeamX.tsv [--test-dir <wav dir>]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

EXPECTED_HEADER = ["filename", "cm-score"]
EXPECTED_ROWS = 1671


def read_tsv(path: Path) -> tuple[list[str], list[list[str]]]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{path}: has a UTF-8 BOM")
    text = raw.decode("utf-8")
    if "\r" in text:
        raise ValueError(f"{path}: contains CR characters (expected LF line endings)")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    rows = [l.split("\t") for l in lines]
    return rows[0], rows[1:]


def validate(pred: Path, template: Path, test_dir: Path | None = None, expected_rows: int | None = EXPECTED_ROWS) -> list[str]:
    errs: list[str] = []
    try:
        th, trows = read_tsv(template)
    except ValueError:
        # the organizer template itself may use CRLF; be lenient for the template only
        t = template.read_text(encoding="utf-8-sig").replace("\r", "").strip("\n").split("\n")
        th, trows = t[0].split("\t"), [l.split("\t") for l in t[1:]]
    try:
        ph, prows = read_tsv(pred)
    except ValueError as e:
        return [str(e)]
    if ph != EXPECTED_HEADER:
        errs.append(f"header {ph!r} != {EXPECTED_HEADER!r}")
    if th != EXPECTED_HEADER:
        errs.append(f"template header unexpected: {th!r}")
    tnames = [r[0] for r in trows]
    if expected_rows is not None and len(tnames) != expected_rows:
        errs.append(f"template has {len(tnames)} rows, expected {expected_rows}")
    bad_cols = [i + 2 for i, r in enumerate(prows) if len(r) != 2]
    if bad_cols:
        errs.append(f"{len(bad_cols)} rows without exactly 2 tab-separated columns (first lines: {bad_cols[:5]})")
    pnames = [r[0] for r in prows if len(r) >= 1]
    if len(prows) != len(trows):
        errs.append(f"row count {len(prows)} != template {len(trows)}")
    dups = {n for n in pnames if pnames.count(n) > 1} if len(pnames) < 5000 else set()
    if dups:
        errs.append(f"{len(dups)} duplicate filenames, e.g. {sorted(dups)[:3]}")
    missing = set(tnames) - set(pnames)
    extra = set(pnames) - set(tnames)
    if missing:
        errs.append(f"{len(missing)} template filenames missing, e.g. {sorted(missing)[:3]}")
    if extra:
        errs.append(f"{len(extra)} filenames not in template, e.g. {sorted(extra)[:3]}")
    if pnames != tnames and not missing and not extra:
        errs.append("row order differs from template (allowed by the scorer, but we preserve template order)")
    nonnum, oob, vals = 0, 0, []
    for r in prows:
        if len(r) != 2:
            continue
        try:
            v = float(r[1])
        except ValueError:
            nonnum += 1
            continue
        vals.append(v)
        if not math.isfinite(v):
            nonnum += 1
        elif not 0.0 <= v <= 1.0:
            oob += 1
    if nonnum:
        errs.append(f"{nonnum} non-numeric / non-finite scores")
    if oob:
        errs.append(f"{oob} scores outside [0,1]")
    if test_dir is not None:
        wavs = {p.name for p in Path(test_dir).glob("*.wav")}
        if wavs != set(tnames):
            errs.append(f"test dir files ({len(wavs)}) != template filenames ({len(tnames)})")
    if vals and len(set(vals)) == 1:
        errs.append("all scores identical (placeholder?)")
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--template", required=True, type=Path)
    ap.add_argument("--test-dir", type=Path)
    a = ap.parse_args(argv)
    errs = validate(a.pred, a.template, a.test_dir)
    if errs:
        print("TSV VALIDATION FAILED:", file=sys.stderr)
        for e in errs:
            print("  -", e, file=sys.stderr)
        return 1
    print(f"OK: {a.pred} — {EXPECTED_ROWS} rows, header {EXPECTED_HEADER}, exact filename set, scores in [0,1]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
