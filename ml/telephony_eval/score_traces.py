#!/usr/bin/env python3
"""Slide the live system's window over every frozen call x channel and save per-hop traces.

Live parameters: 16 kHz, 3.0 s window, 0.5 s hop. A window is scored at every hop end t >= 3.0 s over
audio[t-3s, t]. FloorTracker is updated with every 0.5 s hop as it "arrives" (from t = 0.5 s), and
window_quality uses that call-level floor - identical to api/app/audio/realtime.py usage in the live path.
Logits come from PrimaryDetector.logits (FP16 on CUDA), i.e. the serving code path.

    python ml/telephony_eval/score_traces.py                                  # selected model, all calls
    python ml/telephony_eval/score_traces.py --model-dir models/xlsr12_v4_codec_hard --split eval \
        --out-dir outputs/telephony_eval/xlsr12_v4_codec_hard
Schema: ml/telephony_eval/README.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "ml/telephony_eval"))
sys.path.insert(0, str(ROOT / "api"))
from audio_cache import Cache  # noqa: E402
from channels import CHANNELS, PACKET, apply, render_call  # noqa: E402
from app.audio.realtime import FloorTracker, window_quality  # noqa: E402

SR = 16000
WIN, HOP = 3 * SR, SR // 2
QFIELDS = ["speech_s", "snr_db", "level_dbfs", "clip_ratio", "floor_db"]


def window_label(call: dict, start_ms: int, end_ms: int) -> int:
    if call["kind"] != "mixed":
        return int(call["kind"] == "synthetic")
    if start_ms >= call["switch_ms"]:
        return 1
    return 0 if end_ms <= call["switch_ms"] else -1  # -1: window straddles the switch


def trace_call(call: dict, x: np.ndarray, channel: str):
    """Returns (windows as views, per-window rows). Mirrors the live hop loop."""
    y, lost = apply(channel, x, call["seed"])
    ft = FloorTracker()
    wins, rows = [], []
    for k in range(len(y) // HOP):
        e = (k + 1) * HOP
        ft.update(y[e - HOP:e])
        if e < WIN:
            continue
        w = y[e - WIN:e]
        q = window_quality(w, ft.floor)
        lf = float(lost[(e - WIN) // PACKET:e // PACKET].mean())
        hop_speech = window_quality(y[e - HOP:e], ft.floor)["speech_s"]  # newest 0.5 s hop, same floor
        wins.append(w)
        rows.append([e * 1000 // SR, window_label(call, (e - WIN) * 1000 // SR, e * 1000 // SR)]
                    + [q[f] for f in QFIELDS] + [lf, hop_speech])
    return wins, rows, float(lost.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/selected")
    ap.add_argument("--split", default="all", choices=["all", "calib", "eval"])
    ap.add_argument("--channels", nargs="*", default=CHANNELS)
    ap.add_argument("--out-dir", default="outputs/telephony_eval")
    ap.add_argument("--calls", default="outputs/telephony_eval/calls.json")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=8, help="calls rendered per GPU pass")
    ap.add_argument("--no-floor", action="store_true", help="ablation: no -70 dBFS floor (quality fields then degenerate)")
    ap.add_argument("--limit", type=int, default=None, help="first N calls only (smoke tests)")
    a = ap.parse_args()
    import torch
    from app.detectors.primary import PrimaryDetector

    spec = json.loads((ROOT / a.calls).read_text())
    calls = [c for c in spec["calls"] if a.split == "all" or c["split"] == a.split][: a.limit]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)  # GPU is shared with the backend benchmarks
    det = PrimaryDetector(ROOT / a.model_dir, dev)
    cache = Cache("val")
    acc = {ch: {"rows": [], "logit": [], "call_index": [], "call_loss": []} for ch in a.channels}

    def work(call):
        x = render_call(call, cache, spec["level_dbfs"], -np.inf if a.no_floor else spec["floor_dbfs"])
        return {ch: trace_call(call, x, ch) for ch in a.channels}

    t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        for b in range(0, len(calls), a.chunk):
            chunk = calls[b:b + a.chunk]
            res = list(ex.map(work, chunk))
            wins = [w for r in res for ch in a.channels for w in r[ch][0]]
            lg = det.logits(wins, bs=a.bs)
            o = 0
            for j, r in enumerate(res):
                for ch in a.channels:
                    _, rows, call_loss = r[ch]
                    acc[ch]["rows"] += rows
                    acc[ch]["logit"].append(lg[o:o + len(rows)])
                    acc[ch]["call_index"].append(np.full(len(rows), b + j, np.int32))
                    acc[ch]["call_loss"].append(call_loss)
                    o += len(rows)
            done = b + len(chunk)
            print(f"{done}/{len(calls)} calls, {len(wins)} windows in chunk, {time.time() - t0:.0f}s", flush=True)

    out = ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    meta = {"model_dir": a.model_dir, "model_version": det.version, "calibration": det.cal,
            "calls_seed": spec["seed"], "split": a.split, "no_floor": a.no_floor, "win_ms": 3000, "hop_ms": 500, "sr": SR,
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    for ch, d in acc.items():
        rows = np.array(d["rows"], np.float64)
        np.savez_compressed(
            out / f"traces_{ch}.npz",
            call_index=np.concatenate(d["call_index"]), t_ms=rows[:, 0].astype(np.int32),
            y=rows[:, 1].astype(np.int8), logit=np.concatenate(d["logit"]).astype(np.float32),
            **{f: rows[:, 2 + k].astype(np.float32) for k, f in enumerate(QFIELDS)},
            loss_frac=rows[:, 2 + len(QFIELDS)].astype(np.float32),
            hop_speech_s=rows[:, 3 + len(QFIELDS)].astype(np.float32),
            call_id=np.array([c["call_id"] for c in calls]), call_split=np.array([c["split"] for c in calls]),
            call_kind=np.array([c["kind"] for c in calls]), call_speaker=np.array([c["speaker"] for c in calls]),
            call_generator=np.array([c["generator"] for c in calls]),
            call_duration_ms=np.array([c["duration_ms"] for c in calls], np.int32),
            call_switch_ms=np.array([c["switch_ms"] if c["switch_ms"] is not None else -1 for c in calls], np.int32),
            call_loss_frac=np.array(d["call_loss"], np.float32),
            channel=np.array(ch), meta=np.array(json.dumps(meta)))
        print("wrote", out / f"traces_{ch}.npz", len(rows), "windows")


if __name__ == "__main__":
    main()
