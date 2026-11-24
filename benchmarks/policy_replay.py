"""Call-level replay of the risk policy and the Aegis gate over precomputed traces (PRD §7, §11).

Because the detector is deterministic, replaying per-window logits gives the exact call-level decisions the live
system would make for every policy, without GPU time: false alerts on genuine calls, detection and delay on
synthetic and mixed calls, abstention, and the compute actually spent (windows inferred / hops).

1. Fit: grid-search RiskPolicy (H, persist, min_snr_db) on the `calib` split, telephone-family channels only,
   for the smallest median detection delay subject to genuine-call false-alert rate ≤ --max-fa and
   synthetic-call abstention ≤ --max-abstain. Writes models/selected/risk_policy.json.
2. Evaluate on the speaker-disjoint `eval` split, every channel, fixed-rate vs adaptive gating, with bootstrap
   95 % CIs over calls. Writes outputs/benchmarks/policy_replay.json.

Traces: ml/telephony_eval (see its README for the schema). Simulated calls and channels, not real phone data.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "api"), str(ROOT)]

from engine.aegis import decide  # noqa: E402

from app.risk.engine import Calibration, RiskEngine, RiskPolicy  # noqa: E402

TEL = ("telephone", "telephone_noise30", "telephone_noise20", "telephone_loss5", "opus_telephone")


def load(trace_dir: Path, channel: str) -> dict | None:
    p = trace_dir / f"traces_{channel}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    if "hop_speech_s" not in d:  # older traces: approximate newest-hop speech from the window's speech change
        print(f"[warn] {p.name}: no hop_speech_s, approximating", file=sys.stderr)
        sp, ci = d["speech_s"], d["call_index"]
        prev = np.r_[np.nan, sp[:-1]]
        prev[np.r_[True, ci[1:] != ci[:-1]]] = np.nan
        d["hop_speech_s"] = np.where(np.isnan(prev), np.minimum(0.5, sp / 6), np.clip(sp - prev + 0.25, 0, 0.5))
    starts = np.r_[0, np.nonzero(np.diff(d["call_index"]))[0] + 1, len(d["call_index"])]
    d["_spans"] = {int(d["call_index"][a]): (a, b) for a, b in zip(starts[:-1], starts[1:]) if b > a}
    return d


def replay_call(d: dict, ci: int, cal: Calibration, pol: RiskPolicy, gate: str) -> dict:
    a, b = d["_spans"][ci]
    r = RiskEngine(cal, pol)
    since, inferred, alert_t, alert_speech = 0, 0, None, None
    for i in range(a, b):
        act = decide(gate, float(d["hop_speech_s"][i]), r.state, r.recent_llr, since)
        if act in ("infer", "escalate"):
            since, inferred = 0, inferred + 1
            r.update(float(d["logit"][i]), {"speech_s": float(d["speech_s"][i]), "snr_db": float(d["snr_db"][i]),
                                            "packet_loss": float(d["loss_frac"][i])})
            if r.state == "suspicious" and alert_t is None:
                alert_t, alert_speech = int(d["t_ms"][i]), r.usable_speech_s
        else:
            since += 1
    return {"final": r.state, "alert_ms": alert_t, "alert_speech_s": alert_speech, "inferred": inferred,
            "hops": b - a}


def score(d: dict, calls: list[int], cal: Calibration, pol: RiskPolicy, gate: str) -> list[dict]:
    out = []
    for ci in calls:
        r = replay_call(d, ci, cal, pol, gate)
        r["kind"] = str(d["call_kind"][ci])
        r["switch_ms"] = int(d["call_switch_ms"][ci])
        out.append(r)
    return out


def summarise(rows: list[dict], rng: np.random.Generator, boot: int = 1000) -> dict:
    def rate(flags):
        f = np.asarray(flags, float)
        if not len(f):
            return None, None, None
        bs = rng.choice(f, (boot, len(f))).mean(1)
        return round(float(f.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)

    gen = [r for r in rows if r["kind"] == "genuine"]
    syn = [r for r in rows if r["kind"] == "synthetic"]
    mix = [r for r in rows if r["kind"] == "mixed"]
    fa, fa_lo, fa_hi = rate([r["alert_ms"] is not None for r in gen])
    det, det_lo, det_hi = rate([r["alert_ms"] is not None for r in syn])
    # mixed: an alert before the switch is a false alert; detection = alert at/after the switch
    mix_det, _, _ = rate([r["alert_ms"] is not None and r["alert_ms"] >= r["switch_ms"] for r in mix])
    mix_early, _, _ = rate([r["alert_ms"] is not None and r["alert_ms"] < r["switch_ms"] for r in mix])
    d_syn = [r["alert_ms"] / 1000 for r in syn if r["alert_ms"] is not None]
    d_speech = [r["alert_speech_s"] for r in syn if r["alert_speech_s"] is not None]
    d_mix = [(r["alert_ms"] - r["switch_ms"]) / 1000 for r in mix if r["alert_ms"] is not None and r["alert_ms"] >= r["switch_ms"]]
    abst = lambda rs: round(float(np.mean([r["final"] in ("inconclusive", "analysis_unavailable") and r["alert_ms"] is None
                                           for r in rs])), 4) if rs else None
    inf, hops = sum(r["inferred"] for r in rows), sum(r["hops"] for r in rows)
    pct = lambda xs, q: round(float(np.percentile(xs, q)), 2) if xs else None
    return {"genuine_calls": len(gen), "genuine_false_alert_rate": fa, "fa_ci95": [fa_lo, fa_hi],
            "synthetic_calls": len(syn), "synthetic_detection_rate": det, "det_ci95": [det_lo, det_hi],
            "mixed_calls": len(mix), "mixed_detection_rate": mix_det, "mixed_alert_before_switch_rate": mix_early,
            "first_alert_s_p50": pct(d_syn, 50), "first_alert_s_p95": pct(d_syn, 95),
            "first_alert_usable_speech_s_p50": pct(d_speech, 50), "first_alert_usable_speech_s_p95": pct(d_speech, 95),
            "mixed_delay_after_switch_s_p50": pct(d_mix, 50),
            "genuine_abstain_rate": abst(gen), "synthetic_abstain_rate": abst(syn),
            "windows_inferred": inf, "hops": hops, "compute_fraction": round(inf / hops, 4) if hops else None}


def split_calls(d: dict, split: str) -> list[int]:
    return [ci for ci in d["_spans"] if str(d["call_split"][ci]) == split]


def main(a):
    td = Path(a.traces)
    traces = {ch: t for ch in (*TEL, "clean", "replay_sim") if (t := load(td, ch)) is not None}
    if not traces:
        sys.exit(f"no traces in {td}")
    model_dir = ROOT / "models" / "selected"
    cal = {fam: Calibration.load(model_dir, fam) for fam in ("clean", "telephone")}
    fam = lambda ch: "telephone" if ch in TEL else "clean"
    rng = np.random.default_rng(0)

    # 1. fit on calib, telephone family, fixed-rate gate (the policy must not depend on the scheduler)
    grid = []
    for H, persist, snr in itertools.product(a.H, a.persist, a.min_snr):
        pol = replace(RiskPolicy(), H=H, persist=persist, min_snr_db=snr)
        rows = [r for ch, d in traces.items() if ch in TEL
                for r in score(d, split_calls(d, "calib"), cal["telephone"], pol, "fifo")]
        s = summarise(rows, rng, boot=200)
        grid.append({"H": H, "persist": persist, "min_snr_db": snr, **{k: s[k] for k in (
            "genuine_false_alert_rate", "synthetic_detection_rate", "synthetic_abstain_rate", "first_alert_s_p50",
            "mixed_alert_before_switch_rate")}})
    ok = [g for g in grid if g["genuine_false_alert_rate"] <= a.max_fa and (g["synthetic_abstain_rate"] or 0) <= a.max_abstain
          and g["first_alert_s_p50"] is not None]
    if ok:
        best = min(ok, key=lambda g: (-g["synthetic_detection_rate"], g["first_alert_s_p50"]))
        chosen = "meets constraints on calib"
    else:
        best = min(grid, key=lambda g: (g["genuine_false_alert_rate"], -g["synthetic_detection_rate"]))
        chosen = "NO configuration met the constraints on calib; lowest false-alert configuration chosen"
    pol = replace(RiskPolicy(), H=best["H"], persist=best["persist"], min_snr_db=best["min_snr_db"])
    print("chosen:", chosen, best)
    if a.write_policy:
        (model_dir / "risk_policy.json").write_text(json.dumps(asdict(pol) | {
            "fitted_on": "simulated calls, calib split (speaker-disjoint from eval), telephone-family channels",
            "constraints": {"max_genuine_false_alert_rate": a.max_fa, "max_synthetic_abstain_rate": a.max_abstain},
            "selection": chosen, "created": datetime.now(timezone.utc).isoformat()}, indent=2))

    # 2. evaluate on eval, every channel, both gates
    results = []
    for ch, d in traces.items():
        calls = split_calls(d, "eval")
        for gate in ("fifo", "adaptive"):
            s = summarise(score(d, calls, cal[fam(ch)], pol, gate), rng)
            results.append({"channel": ch, "gating": "fixed_rate" if gate == "fifo" else "adaptive", **s})
            print(ch, gate, {k: s[k] for k in ("genuine_false_alert_rate", "synthetic_detection_rate",
                                               "first_alert_s_p50", "compute_fraction")}, flush=True)
    reduction = []
    for ch in traces:
        f, ad = [next(r for r in results if r["channel"] == ch and r["gating"] == g) for g in ("fixed_rate", "adaptive")]
        reduction.append({"channel": ch, "compute_reduction": round(1 - ad["windows_inferred"] / f["windows_inferred"], 4),
                          "delta_false_alert": round(ad["genuine_false_alert_rate"] - f["genuine_false_alert_rate"], 4),
                          "delta_detection": round(ad["synthetic_detection_rate"] - f["synthetic_detection_rate"], 4),
                          "delta_first_alert_s_p50": None if None in (ad["first_alert_s_p50"], f["first_alert_s_p50"])
                          else round(ad["first_alert_s_p50"] - f["first_alert_s_p50"], 2)})
    out = {"created": datetime.now(timezone.utc).isoformat(), "kind": "policy_replay",
           "description": ("Exact call-level replay of the risk policy + Aegis gate over per-window traces of the "
                           "serving model on SIMULATED calls/channels (ml/telephony_eval). Policy fitted on the calib "
                           "split; all numbers below are on the speaker-disjoint eval split."),
           "risk_policy": asdict(pol), "selection": chosen, "calibration": {k: asdict(v) for k, v in cal.items()},
           "results": results, "adaptive_vs_fixed": reduction, "fit_grid": grid}
    p = ROOT / "outputs" / "benchmarks" / "policy_replay.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=float))
    print("wrote", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", default=str(ROOT / "outputs" / "telephony_eval"))
    ap.add_argument("--H", type=float, nargs="+", default=[1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
    ap.add_argument("--persist", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--min-snr", type=float, nargs="+", default=[8.0, 12.0, 15.0])
    ap.add_argument("--max-fa", type=float, default=0.01)
    ap.add_argument("--max-abstain", type=float, default=0.10)
    ap.add_argument("--write-policy", action="store_true")
    main(ap.parse_args())
