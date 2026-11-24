"""Two-stage screening replay: distilled student first, 12-layer model only when the student is not confidently
clear (PRD §7 "lightweight scoring stage"). Exact call-level replay over the eval-split traces of both models.

    .venv/Scripts/python.exe benchmarks/cascade_replay.py --student models/exp_student_l4 --cost-ratio 0.45

Compute is reported in main-model-equivalents: cost = n_main + cost_ratio * n_student, where cost_ratio is the
student/main per-window latency ratio measured on this GPU (--cost-ratio). Writes outputs/benchmarks/cascade_replay.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "api"), str(ROOT), str(ROOT / "benchmarks")]

from engine.aegis import decide  # noqa: E402
from policy_replay import TEL, load, split_calls, summarise  # noqa: E402

from app.risk.engine import Calibration, RiskEngine, RiskPolicy  # noqa: E402


def student_index(d: dict) -> dict:
    ids = d["call_id"]
    return {(str(ids[c]), int(t)): float(lg) for c, t, lg in zip(d["call_index"], d["t_ms"], d["logit"])}


def replay(d, ci, s_idx, cal, s_cal, pol, clear_llr) -> dict:
    a, b = d["_spans"][ci]
    cid = str(d["call_id"][ci])
    r = RiskEngine(cal, pol)
    since, n_main, n_student, alert = 0, 0, 0, None
    for i in range(a, b):
        act = decide("adaptive", float(d["hop_speech_s"][i]), r.state, r.recent_llr, since)
        if act not in ("infer", "escalate"):
            since += 1
            continue
        since = 0
        q = {"speech_s": float(d["speech_s"][i]), "snr_db": float(d["snr_db"][i]), "packet_loss": float(d["loss_frac"][i])}
        s = s_idx.get((cid, int(d["t_ms"][i]))) if clear_llr is not None else None
        if act == "infer" and s is not None:
            n_student += 1
            if s_cal.llr(s) < clear_llr:
                r.update(s, q, cal=s_cal)
            else:
                n_main += 1
                r.update(float(d["logit"][i]), q)
        else:
            n_main += 1
            r.update(float(d["logit"][i]), q)
        if r.state == "suspicious" and alert is None:
            alert = (int(d["t_ms"][i]), r.usable_speech_s)
    return {"final": r.state, "alert_ms": alert and alert[0], "alert_speech_s": alert and alert[1],
            "n_main": n_main, "n_student": n_student, "hops": b - a, "kind": str(d["call_kind"][ci]),
            "switch_ms": int(d["call_switch_ms"][ci])}


def main(a):
    td, sd = Path(a.traces), Path(a.traces) / Path(a.student).name
    model_dir = ROOT / "models" / "selected"
    pol = RiskPolicy.load(model_dir)
    s_cal = Calibration.load(ROOT / a.student, "telephone")
    rng = np.random.default_rng(0)
    rows = []
    for ch in (*TEL, "clean"):
        d, sdat = load(td, ch), load(sd, ch)
        if d is None or sdat is None:
            continue
        cal = Calibration.load(model_dir, "telephone" if ch in TEL else "clean")
        s_idx = student_index(sdat)
        calls = split_calls(d, "eval")
        for clear in [None, *a.clear_llr]:
            res = [replay(d, ci, s_idx, cal, s_cal, pol, clear) for ci in calls]
            s = summarise([{**r, "inferred": r["n_main"]} for r in res], rng, boot=500)
            hops = sum(r["hops"] for r in res)
            cost = sum(r["n_main"] for r in res) + a.cost_ratio * sum(r["n_student"] for r in res)
            rows.append({"channel": ch, "screen_clear_llr": clear if clear is not None else "off (adaptive only)",
                         "genuine_false_alert_rate": s["genuine_false_alert_rate"],
                         "synthetic_detection_rate": s["synthetic_detection_rate"],
                         "mixed_detection_rate": s["mixed_detection_rate"],
                         "first_alert_s_p50": s["first_alert_s_p50"], "first_alert_s_p95": s["first_alert_s_p95"],
                         "main_windows": sum(r["n_main"] for r in res), "student_windows": sum(r["n_student"] for r in res),
                         "cost_vs_fixed_rate": round(cost / hops, 4)})
            print(json.dumps(rows[-1]), flush=True)
    out = {"created": datetime.now(timezone.utc).isoformat(), "kind": "cascade_replay",
           "description": ("Two-stage screening replay on SIMULATED eval calls: adaptive gate, then the distilled "
                           "student; windows the student scores confidently clear skip the 12-layer model. Cost in "
                           "main-model equivalents relative to fixed-rate inference on every hop."),
           "student": a.student, "student_calibration": asdict(s_cal), "cost_ratio": a.cost_ratio,
           "risk_policy": asdict(pol), "results": rows}
    p = ROOT / "outputs" / "benchmarks" / "cascade_replay.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print("wrote", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", default=str(ROOT / "outputs" / "telephony_eval"))
    ap.add_argument("--student", default="models/exp_student_l4")
    ap.add_argument("--cost-ratio", type=float, required=True, help="student / main per-window latency (measured)")
    ap.add_argument("--clear-llr", type=float, nargs="+", default=[-4.0, -3.0, -2.0, -1.0, 0.0])
    main(ap.parse_args())
