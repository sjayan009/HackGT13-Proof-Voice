"""Deterministic evidence-based orchestration (NSA "choose analyses by context").

    quality scan (always, cheap) ─┐
    metadata (if a file path)     ├─> primary detector (always) + sliding-window timeline
                                  │
    if lossy codec / band-limited ──────────────> compression branch
    if primary decision is near threshold ──────> spectral + prosody branches
    if timeline shows a local jump/anomaly ─────> splice branch on the whole clip, regions reported
    ↓
    status + analysis_confidence + suspicious regions + time-to-confidence

Only the primary detector sets `cm_score`; hand-crafted branches enter the score only if fusion is shown to
improve validation minDCF (see RESULTS.md / outputs/results/fusion.json). Otherwise every other branch is reported
as supporting evidence with `used_in_score: false`.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np

from app.forensics import compression, prosody, quality, spectral, splice
from app.forensics import metadata as meta_branch

SR = 16000
DISCLAIMER = ("Probabilistic evidence, not proof. The score comes from a detector trained on specific generators and "
              "may not generalize to unseen synthesis methods, heavy compression, or replayed audio. Metadata can be "
              "forged or stripped. Use alongside other verification.")
MIN_SPEECH_S = 1.0
UNCERTAIN_BAND = 0.25  # |p - threshold| below this -> "weak" primary decision -> deeper analysis


def status_for(p: float, thr: float, speech_s: float, conf: float) -> str:
    if speech_s < MIN_SPEECH_S:
        return "insufficient_evidence"
    lo, hi = thr * 0.5, thr + (1 - thr) * 0.35
    if p >= max(hi, thr):
        return "likely_synthetic"
    if p <= lo and conf >= 0.4:
        return "likely_human"
    return "inconclusive"


def analysis_confidence(p: float, thr: float, q: dict, windows: list[dict]) -> float:
    """Evidence sufficiency in [0,1]: speech duration, SNR, decisiveness, window agreement. Heuristic, documented."""
    speech_s = q.get("speech_ratio", 1.0) * q.get("duration_s", 0.0)
    dur = min(1.0, speech_s / 3.0)
    snr = float(np.clip((q.get("snr_estimate_db", 30.0) - 5.0) / 20.0, 0.0, 1.0))
    margin = abs(p - thr) / max(thr, 1 - thr)
    decisive = float(np.clip(margin * 1.5, 0.0, 1.0))
    if len(windows) > 1:
        wp = np.array([w["synthetic_probability"] for w in windows])
        agree = float(np.mean((wp >= thr) == (p >= thr)))
    else:
        agree = 0.7
    return round(float(0.3 * dur + 0.15 * snr + 0.35 * decisive + 0.2 * agree), 3)


def suspicious_regions(windows: list[dict], thr: float) -> list[dict]:
    regs, cur = [], None
    for w in windows:
        if w["synthetic_probability"] >= thr:
            if cur and w["start_ms"] <= cur["end_ms"]:
                cur["end_ms"] = max(cur["end_ms"], w["end_ms"])
                cur["peak_probability"] = max(cur["peak_probability"], w["synthetic_probability"])
            else:
                cur = {"start_ms": w["start_ms"], "end_ms": w["end_ms"], "peak_probability": w["synthetic_probability"],
                       "reason": "detector window score above decision threshold"}
                regs.append(cur)
        else:
            cur = None
    for r in regs:
        r["peak_probability"] = round(float(r["peak_probability"]), 4)
    return regs


def time_to_confidence(windows: list[dict], thr: float, final_p: float) -> int | None:
    """First window end time from which every subsequent window agrees with the final decision."""
    if not windows:
        return None
    final = final_p >= thr
    t = None
    for w in reversed(windows):
        if (w["synthetic_probability"] >= thr) == final:
            t = w["end_ms"]
        else:
            break
    return t


def analyze(audio: np.ndarray, detector, path: str | None = None, file_name: str | None = None,
            orig_sr: int | None = None, win_ms: int = 2000, hop_ms: int = 500) -> dict:
    t_all = time.perf_counter()
    techniques, why, evidence, log = [], {}, {}, []

    def run(name: str, reason: str, fn):
        t = time.perf_counter()
        try:
            ev = fn()
        except Exception as e:  # noqa: BLE001 - a failing branch must never kill the report
            ev = {"summary": f"branch failed: {type(e).__name__}", "suspicion": None, "features": {}, "flags": [],
                  "used_in_score": False}
        ms = round((time.perf_counter() - t) * 1000, 2)
        techniques.append(name)
        why[name] = reason
        evidence[name] = ev
        log.append({"technique": name, "ran": True, "reason": reason, "runtime_ms": ms})
        return ev

    def skip(name: str, reason: str):
        log.append({"technique": name, "ran": False, "reason": reason, "runtime_ms": 0.0})

    # 1. file / container
    file_info = {"name": file_name, "container": None, "codec": None, "sample_rate": orig_sr, "channels": None,
                 "bit_depth": None, "duration_s": round(len(audio) / SR, 3), "bitrate": None, "metadata": {}}
    lossy = False
    if path:
        info = meta_branch.inspect_file(path)
        file_info.update({k: v for k, v in info.items() if v not in (None, {})})
        file_info["name"] = file_name
        ev = run("metadata", "file provided: container/codec/tag inspection", lambda: meta_branch.analyze_metadata(info))
        lossy = bool(ev["features"].get("is_lossy_codec"))
    else:
        skip("metadata", "live stream: no container to inspect")

    # 2. quality (routing + confidence)
    q_ev = run("quality", "always: signal quality drives routing and confidence", lambda: quality.analyze(audio, SR))
    q = q_ev["features"]

    # 3. primary detector
    t = time.perf_counter()
    res = detector.score(audio, SR)
    windows = detector.score_windows(audio, win_ms, hop_ms) if len(audio) > int(1.25 * win_ms * SR / 1000) else []
    thr = detector.threshold
    p = res.synthetic_probability
    det_ms = round((time.perf_counter() - t) * 1000, 2)
    techniques.append("primary_detector")
    why["primary_detector"] = "always: learned anti-spoofing model (sets the score)"
    wp = [w["synthetic_probability"] for w in windows]
    evidence["primary_detector"] = {
        "summary": (f"Learned detector gives synthetic probability {p:.2f} (decision threshold {thr:.2f}, chosen to "
                    f"minimise the official minDCF on validation)."),
        "suspicion": round(p, 4),
        "features": {"logit": round(res.raw_score, 4), "window_min": round(min(wp), 4) if wp else None,
                     "window_max": round(max(wp), 4) if wp else None,
                     "window_std": round(float(np.std(wp)), 4) if wp else None},
        "flags": [], "used_in_score": True}
    log.append({"technique": "primary_detector", "ran": True, "reason": why["primary_detector"], "runtime_ms": det_ms})

    # 4. context-routed branches
    cutoff_hint = None
    weak = abs(p - thr) < UNCERTAIN_BAND
    band_check = spectral.extract_features(audio, SR) if not weak else None
    if band_check is not None:
        cutoff_hint = band_check.get("band_limit_cutoff_hz")
    band_limited = cutoff_hint is not None and cutoff_hint < 6500
    if weak:
        run("spectral", f"primary decision weak (|p-thr|={abs(p - thr):.2f} < {UNCERTAIN_BAND})",
            lambda: spectral.analyze(audio, SR))
        run("prosody", "primary decision weak: check pitch/pause/energy dynamics", lambda: prosody.analyze(audio, SR))
        cutoff_hint = evidence["spectral"]["features"].get("band_limit_cutoff_hz")
        band_limited = cutoff_hint is not None and cutoff_hint < 6500
    else:
        skip("spectral", f"primary decision confident (|p-thr|={abs(p - thr):.2f})")
        skip("prosody", "primary decision confident")
    if lossy or band_limited:
        run("compression", "lossy codec in container" if lossy else f"band-limited spectrum (~{cutoff_hint:.0f} Hz)",
            lambda: compression.analyze(audio, SR))
    else:
        skip("compression", "lossless container and full-band spectrum")
    local_jump = len(wp) > 2 and (max(wp) - min(wp) > 0.4 or float(np.max(np.abs(np.diff(wp)))) > 0.3)
    if local_jump:
        run("splice", f"detector timeline jumps (range {max(wp) - min(wp):.2f}): look for edit seams",
            lambda: splice.analyze(audio, SR))
    else:
        skip("splice", "timeline homogeneous" if wp else "clip too short for a timeline")

    conf = analysis_confidence(p, thr, q, windows)
    speech_s = q.get("speech_ratio", 1.0) * q.get("duration_s", len(audio) / SR)
    regions = suspicious_regions(windows, thr)
    if "splice" in evidence:
        for r in evidence["splice"].get("regions", []):
            regions.append({**r, "peak_probability": None})
    return {
        "id": str(uuid.uuid4()),
        "file": file_info,
        "synthetic_probability": round(p, 4),
        "cm_score": round(p, 6),
        "analysis_confidence": conf,
        "status": status_for(p, thr, speech_s, conf),
        "decision_threshold": round(thr, 4),
        "detector": {"name": detector.name, "version": detector.version, "raw_score": round(res.raw_score, 4),
                     "latency_ms": round(res.latency_ms, 2)},
        "timeline": [{"start_ms": w["start_ms"], "end_ms": w["end_ms"],
                      "synthetic_probability": round(w["synthetic_probability"], 4),
                      "analysis_confidence": None} for w in windows],
        "suspicious_regions": regions,
        "time_to_confidence_ms": time_to_confidence(windows, thr, p),
        "techniques_run": techniques,
        "why_run": why,
        "evidence": evidence,
        "branch_log": log,
        "processing_ms": round((time.perf_counter() - t_all) * 1000, 2),
        "disclaimer": DISCLAIMER,
    }
