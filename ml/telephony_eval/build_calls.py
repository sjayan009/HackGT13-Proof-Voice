#!/usr/bin/env python3
"""Freeze a call-level evaluation set built from the validation manifest (manifest only, no audio written).

A simulated "call" is 20-40 s of one voice: validation clips concatenated with 0.3-1.0 s pauses.
  genuine   : clips of one bona fide speaker
  synthetic : clips of one generator + one cloned speaker
  mixed     : 8-16 s of a bona fide speaker, then synthetic speech from one generator (switch_ms recorded).
              LJ mixed calls switch to the LJ voice (same claimed speaker); LibriSpeech mixed calls switch
              to the cloned speaker of the same split (a different voice: a weaker, easier test).
Every clip is used at most once across all calls.

Speaker-disjoint split (calibration is never fitted on eval voices):
  bona fide LibriSpeech speakers: seeded 8 / 8 split
  speaker_6167 clones -> calib; speaker_2061 clones -> eval (the documented hard speaker stays in eval)
  LJ (bona fide + the 4 single-speaker LJ-voice generators) -> eval
Consequence: calib contains no single-speaker generators and only the easier cloned voice; see README.

    python ml/telephony_eval/build_calls.py      # -> outputs/telephony_eval/calls.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
from common import manifest  # noqa: E402

SEED = 20261124
SR = 16000
MAX_CLIP_S = 14.0  # outputs/cache caps clips here; durations below mirror what the cache returns
CALL_MIN_S, CALL_MAX_S = 20.0, 40.0
GAP_S = (0.3, 1.0)
MIXED_GENUINE_S = (8.0, 16.0)
LEVEL_DBFS = -26.0  # every clip is scaled to this RMS (telephone nominal level), both classes
FLOOR_DBFS = -70.0  # continuous white floor so silence is never digital zero (VAD/SNR stay meaningful)
CLONE_SPLIT = {"speaker_6167": "calib", "speaker_2061": "eval"}
N_MIXED = {"libri": 3, "LJ": 4}  # mixed calls per bona fide speaker
N_SYNTH = {"calib": 20, "eval_clone": 15, "eval_lj": 10}  # synthetic calls per generator+speaker
OUT = ROOT / "outputs/telephony_eval/calls.json"


def pack(pool: list[int], dur: np.ndarray, rng, target_s: float, t0_s: float = 0.0) -> tuple[list[dict], float]:
    """Pop clips from pool until the segment reaches target_s. Returns segments (call-relative) and end time."""
    segs, t = [], t0_s
    while pool and t - t0_s < target_s:
        i = pool.pop()
        if segs or t0_s > 0:
            t += rng.uniform(*GAP_S)
        d = min(dur[i], MAX_CLIP_S)
        segs.append({"val_idx": int(i), "start_ms": int(round(t * 1000)), "dur_ms": int(round(d * 1000))})
        t += d
    return segs, t


def finish(segs: list[dict], max_s: float = CALL_MAX_S) -> int:
    """Drop segments starting past max_s (a pause can push one there), truncate the last; return duration ms."""
    while segs[-1]["start_ms"] >= max_s * 1000 - 500:  # keep >= 0.5 s of the last clip
        segs.pop()
    end = segs[-1]["start_ms"] + segs[-1]["dur_ms"]
    if end > max_s * 1000:
        segs[-1]["dur_ms"] -= end - int(max_s * 1000)
        end = int(max_s * 1000)
    return end


def build() -> dict:
    m = manifest("val")
    rng = np.random.default_rng(SEED)
    dur = m.duration.to_numpy()
    y = m.y.to_numpy()

    libri = sorted(m[(m.y == 0) & (m.source == "librispeech")].speaker.unique())
    perm = rng.permutation(len(libri))
    split_of = {libri[k]: ("calib" if j < len(libri) // 2 else "eval") for j, k in enumerate(perm)}
    split_of["LJ"] = "eval"
    split_of["single"] = "eval"  # LJ-voice spoofs share the LJ speaker
    split_of |= CLONE_SPLIT

    def shuffled(mask) -> list[int]:
        idx = np.flatnonzero(mask)
        return [int(i) for i in idx[rng.permutation(len(idx))]]

    spoof_pools = {(g, s): shuffled((m.generator == g) & (m.speaker == s))
                   for g, s in sorted(m[m.y == 1].groupby(["generator", "speaker"]).groups)}
    clone_gens = sorted({g for g, s in spoof_pools if s != "single"})
    lj_gens = sorted({g for g, s in spoof_pools if s == "single"})

    calls = []

    def add(kind, split, speaker, generator, segs, switch_ms=None):
        calls.append({"call_id": f"{split}_{kind}_{len(calls):04d}", "split": split, "kind": kind,
                      "speaker": speaker, "generator": generator, "seed": SEED + len(calls),
                      "duration_ms": finish(segs), "switch_ms": switch_ms,
                      "segments": [s | {"y": int(y[s["val_idx"]])} for s in segs]})

    for spk in libri + ["LJ"]:
        split = split_of[spk]
        pool = shuffled((m.speaker == spk) & (y == 0))
        n_mixed = N_MIXED["LJ" if spk == "LJ" else "libri"]
        for k in range(n_mixed):
            gsegs, t = pack(pool, dur, rng, rng.uniform(*MIXED_GENUINE_S))
            if spk == "LJ":
                key = (lj_gens[k % len(lj_gens)], "single")
            else:
                clone = next(s for s, sp in CLONE_SPLIT.items() if sp == split)
                key = (clone_gens[(len(calls) + k) % len(clone_gens)], clone)
            ssegs, _ = pack(spoof_pools[key], dur, rng, rng.uniform(CALL_MIN_S, CALL_MAX_S) - t, t)
            add("mixed", split, spk, key[0], gsegs + ssegs, switch_ms=ssegs[0]["start_ms"])
        while True:
            segs, t = pack(pool, dur, rng, rng.uniform(CALL_MIN_S, CALL_MAX_S))
            if t < CALL_MIN_S:  # leftover too short for a call: dropped
                break
            add("genuine", split, spk, m.generator[segs[0]["val_idx"]], segs)

    for (g, s), pool in spoof_pools.items():
        split = split_of[s]
        n = N_SYNTH["calib"] if split == "calib" else N_SYNTH["eval_lj" if s == "single" else "eval_clone"]
        for _ in range(n):
            segs, _ = pack(pool, dur, rng, rng.uniform(CALL_MIN_S, CALL_MAX_S))
            add("synthetic", split, "LJ" if s == "single" else s, g, segs)

    used = [s["val_idx"] for c in calls for s in c["segments"]]
    assert len(used) == len(set(used)), "a clip was reused"
    spk_by_split = {sp: {c["speaker"] for c in calls if c["split"] == sp} for sp in ("calib", "eval")}
    assert not spk_by_split["calib"] & spk_by_split["eval"], "speaker leak between calib and eval"
    for c in calls:
        assert CALL_MIN_S * 1000 <= c["duration_ms"] <= CALL_MAX_S * 1000, c["call_id"]
        assert all(s["dur_ms"] >= 500 and s["start_ms"] + s["dur_ms"] <= c["duration_ms"] for s in c["segments"])
        assert c["kind"] != "mixed" or any(s["y"] == 1 for s in c["segments"]), c["call_id"]
    summary = {sp: {k: sum(1 for c in calls if c["split"] == sp and c["kind"] == k)
                    for k in ("genuine", "synthetic", "mixed")} for sp in ("calib", "eval")}
    return {"seed": SEED, "source_manifest": "outputs/manifests/val.csv", "sr": SR,
            "level_dbfs": LEVEL_DBFS, "floor_dbfs": FLOOR_DBFS, "gap_s": GAP_S, "max_clip_s": MAX_CLIP_S,
            "speaker_split": {k: v for k, v in sorted(split_of.items())}, "summary": summary,
            "total_audio_s": round(sum(c["duration_ms"] for c in calls) / 1000), "calls": calls}


def main():
    out = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("summary", "total_audio_s", "speaker_split")}, indent=1))


if __name__ == "__main__":
    main()
