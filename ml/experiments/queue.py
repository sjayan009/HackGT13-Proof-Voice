#!/usr/bin/env python3
"""Sequential, resumable Workstream-A training queue (PRD section 6). Prepared, not launched automatically.

Every run = the v3 recipe read from models/selected/meta.json + ONE controlled change. After training, each run
is evaluated with calibrate.py (writes the run's own calibration.json; never selects), robustness.py, the
telephony traces/benchmark on the eval half, and unseen.py for leave-one-generator-out runs.

Resumable at step granularity: a step is skipped when its output exists; a run is skipped when
outputs/experiments/<run>.done exists. An interrupted *training* step restarts that run from scratch
(train.py has no mid-run checkpoint resume).

    python ml/experiments/queue.py --dry-run          # commands + wall-clock estimate, runs nothing
    python ml/experiments/queue.py                    # run everything still pending, in priority order
    python ml/experiments/queue.py --only student_l4  # subset
    python ml/experiments/queue.py --smoke codec_only_s0   # 5-step training into models/_smoke_*, no evals
Logs: outputs/experiments/<run>.log
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
LOGS = ROOT / "outputs/experiments"

# (name, controlled change vs v3, kind). Priority order.
RUNS = [
    ("codec_only_s0", ["--codec-prob", "0.5", "--seed", "0"], "codec"),
    ("codec_only_s1", ["--codec-prob", "0.5", "--seed", "1"], "codec"),
    ("student_l4", ["--num-layers", "4", "--teacher", "models/selected", "--kd-weight", "0.5", "--kd-temp", "2.0"],
     "student"),
    ("logo_elevenlabs", ["--exclude-generators", "elevenlabs"], "plain"),
    ("logo_playht", ["--exclude-generators", "playht"], "plain"),
]


def v3_args() -> list[str]:
    a = json.loads((ROOT / "models/selected/meta.json").read_text())["args"]
    out = []
    for k, v in a.items():
        if k in ("out", "exclude_generators", "grad_ckpt", "seed") or v is None or v == []:
            continue
        flag = "--" + k.replace("_", "-")
        out += [flag, *map(str, v)] if isinstance(v, list) else [flag, str(v)]
    return out


def merge(base: list[str], change: list[str]) -> list[str]:
    """Override flags in base with those in change."""
    keys, out, i = {c for c in change if c.startswith("--")}, [], 0
    while i < len(base):
        j = i + 1
        while j < len(base) and not base[j].startswith("--"):
            j += 1
        if base[i] not in keys:
            out += base[i:j]
        i = j
    return out + change


def steps_for(name: str, change: list[str], smoke: bool = False) -> list[tuple[str, list[str], Path]]:
    """[(label, argv, output that marks the step done)]"""
    md = f"models/{'_smoke_' if smoke else 'exp_'}{name}"
    train = [PY, "ml/train.py", *merge(v3_args(), change), "--out", md]
    if smoke:
        return [("train", merge(train[2:], ["--steps", "5", "--eval-every", "5"]), ROOT / md / "val_report.json")]
    tag = Path(md).name
    st = [("train", train, ROOT / md / "val_report.json"),
          ("calibrate", [PY, "ml/calibrate.py", "--model-dir", md], ROOT / md / "val_report_serving.json"),
          ("robustness", [PY, "ml/robustness.py", "--model-dir", md, "--bs", "8",
                          "--output", f"outputs/results/robustness_{tag}.json"],
           ROOT / f"outputs/results/robustness_{tag}.json"),
          ("traces", [PY, "ml/telephony_eval/score_traces.py", "--model-dir", md, "--split", "eval",
                      "--out-dir", f"outputs/telephony_eval/{tag}"],
           ROOT / f"outputs/telephony_eval/{tag}/traces_replay_sim.npz"),
          ("telephony", [PY, "ml/telephony_eval/analyze.py", "--traces", f"outputs/telephony_eval/{tag}",
                         "--output", f"outputs/results/telephony_eval_{tag}.json"],
           ROOT / f"outputs/results/telephony_eval_{tag}.json")]
    if "--exclude-generators" in change:
        st.append(("unseen", [PY, "ml/unseen.py", "--exp", md, "--ref", "models/selected"],
                   ROOT / f"outputs/results/unseen_{tag}.json"))
    return st


def s_per_step(log: str) -> tuple[float, float]:
    """Steady-state seconds/step (median, after warm-up) and mean seconds per full validation pass."""
    lines = (ROOT / log).read_text(errors="ignore").splitlines()
    pts = [(int(m[1]), float(m[2])) for m in (re.match(r"step (\d+) loss \S+ (\d+)s", ln) for ln in lines) if m]
    st = np.array(pts)
    d = np.diff(st, axis=0)
    rate = float(np.median(d[st[1:, 0] > 150][:, 1] / d[st[1:, 0] > 150][:, 0]))
    evals, last = [], None
    for ln in lines:
        m = re.match(r"step (\d+) loss \S+ (\d+)s", ln)
        if m:
            last = float(m[2])
        e = re.search(r"'elapsed_s': (\d+)", ln)
        if e and last is not None:
            evals.append(float(e[1]) - last)
    return rate, float(np.median(evals))


POST_MIN = {"calibrate": 4, "robustness": 20, "traces": 20, "telephony": 3, "unseen": 1}  # rough, minutes
CODEC_PARALLEL = 0.57  # measured: 4 parallel ffmpeg round trips take 0.57x the serial time (train.py now threads them)
STUDENT_FACTOR = 0.6  # GUESS: 4-layer student (~0.45x of a 12-layer step) + frozen 12-layer teacher forward


def estimate(name: str, change: list[str], kind: str) -> float:
    plain, ev = s_per_step("outputs/train_xlsr12_v3.log")
    serial, _ = s_per_step("outputs/train_xlsr12_v4_codec_hard.log")  # v4: --codec-prob 0.5, serial ffmpeg
    codec = plain + (serial - plain) * CODEC_PARALLEL
    args = merge(v3_args(), change)
    steps, every = int(args[args.index("--steps") + 1]), int(args[args.index("--eval-every") + 1])
    rate = {"codec": codec, "plain": plain, "student": plain * STUDENT_FACTOR}[kind]
    ev_s = ev * (0.5 if kind == "student" else 1.0)
    post = sum(POST_MIN[s] for s, _, _ in steps_for(name, change)[1:])
    return (steps * rate + (steps // every) * ev_s) / 3600 + post / 60


def run_step(log: Path, argv: list[str]) -> None:
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} $ {' '.join(argv)}\n")
        f.flush()
        r = subprocess.run(argv, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"step failed ({r.returncode}); see {log}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--smoke", default=None, help="run NAME for 5 steps into models/_smoke_NAME (no evals)")
    a = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    plain, ev = s_per_step("outputs/train_xlsr12_v3.log")
    codec, _ = s_per_step("outputs/train_xlsr12_v4_codec_hard.log")
    print(f"measured: v3 recipe {plain:.2f} s/step, codec-prob 0.5 {codec:.2f} s/step with serial ffmpeg "
          f"(now threaded: ~{plain + (codec - plain) * CODEC_PARALLEL:.2f} s/step), validation pass {ev:.0f} s")

    if a.smoke:
        name, change, _ = next(r for r in RUNS if r[0] == a.smoke)
        label, argv, _ = steps_for(name, change, smoke=True)[0]
        argv = [PY, "ml/train.py", *argv]
        print("SMOKE:", " ".join(argv))
        run_step(LOGS / f"_smoke_{name}.log", argv)
        print("smoke ok; log", LOGS / f"_smoke_{name}.log")
        return

    total = 0.0
    for name, change, kind in RUNS:
        if a.only and name not in a.only:
            continue
        done = LOGS / f"{name}.done"
        h = estimate(name, change, kind)
        total += 0 if done.exists() else h
        print(f"\n[{name}] est {h:.1f} h{' (done, skipped)' if done.exists() else ''}")
        if done.exists():
            continue
        for label, argv, marker in steps_for(name, change):
            state = "skip (exists)" if marker.exists() else ("would run" if a.dry_run else "run")
            print(f"  {label:10s} {state}: {' '.join(argv[1:])}")
            if marker.exists() or a.dry_run:
                continue
            t0 = time.time()
            run_step(LOGS / f"{name}.log", argv)
            print(f"  {label} finished in {(time.time() - t0) / 60:.1f} min", flush=True)
        if not a.dry_run:
            done.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"\nestimated wall-clock for pending runs: {total:.1f} h (single RTX 4060 Laptop, nothing else on the GPU)")


if __name__ == "__main__":
    main()
