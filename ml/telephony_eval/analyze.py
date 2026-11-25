#!/usr/bin/env python3
"""Window-level telephony benchmark, channel-conditioned calibration and quality-gate abstention from traces.

    python ml/telephony_eval/analyze.py --traces outputs/telephony_eval outputs/telephony_eval/xlsr12_v4_codec_hard \
        --write-calibration                     # -> outputs/results/telephony_eval.json + models/selected/channel_calibration.json

* Benchmark: eval split only, windows labelled 0/1 (mixed-call windows straddling the switch are excluded).
  EER, AUC, minDCF (HEARSAY costs: Pspoof 0.5, Cfa 4), genuine false-alarm / spoof miss at the model's own
  serving threshold (calibration.json, fitted on clean validation clips). 95 % CIs: bootstrap over calls,
  stratified by call kind.
* Calibration (first traces dir only, needs the calib split): Platt (a, b) + minDCF threshold per channel family,
  fitted on calib windows with speech_s >= MIN_SPEECH_FIT; temperature scaling of the serving logit for comparison.
* Abstention: risk-coverage of quality gates (speech_s, snr_db, loss_frac) at the serving threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "ml/telephony_eval"))
from channels import CHANNELS, FAMILY  # noqa: E402
from common import save_json  # noqa: E402
from evaluate_mindcf import fast_mindcf  # noqa: E402

N_BOOT = 1000
MIN_SPEECH_FIT = 1.0
GATES = {"speech_s": (">=", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]),
         "snr_db": (">=", [-1e9, 5, 10, 15, 20, 25, 30]),
         "loss_frac": ("<=", [1.0, 0.2, 0.1, 0.05, 0.02, 0.0])}


def sig(z):
    return 1 / (1 + np.exp(-np.asarray(z, np.float64)))


def load(d: Path, ch: str) -> dict:
    z = np.load(d / f"traces_{ch}.npz")
    t = {k: z[k] for k in z.files}
    t["meta"] = json.loads(str(t["meta"]))
    t["win_split"] = t["call_split"][t["call_index"]]
    return t


def present(d: Path) -> list[str]:
    return [c for c in CHANNELS if (d / f"traces_{c}.npz").exists()]


def eer(y, s) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y, s)
    i = int(np.argmin(np.abs(1 - tpr - fpr)))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def metrics(y, logit, p_serv, thr, full=True) -> dict:
    from sklearn.metrics import roc_auc_score

    pred = p_serv >= thr
    fa, miss = float(pred[y == 0].mean()), float((~pred[y == 1]).mean())
    out = {"EER": eer(y, logit), "genuine_FA_at_serving_thr": fa, "spoof_miss_at_serving_thr": miss}
    if full:
        out |= {"AUC": float(roc_auc_score(y, logit)), "minDCF": fast_mindcf(y, sig(logit))[0],
                "actDCF_at_serving_thr": fa + 4 * miss}
    return out


def bootstrap(t, sel, y, logit, p_serv, thr, rng) -> dict:
    """95 % CI over calls (stratified by call kind) for EER and the threshold rates."""
    ci = t["call_index"][sel]
    order = np.argsort(ci, kind="stable")
    calls, starts = np.unique(ci[order], return_index=True)
    groups = np.split(order, starts[1:])
    kinds = t["call_kind"][calls]
    draws = []
    for _ in range(N_BOOT):
        pick = np.concatenate([rng.choice(np.flatnonzero(kinds == k), (kinds == k).sum()) for k in np.unique(kinds)])
        ix = np.concatenate([groups[i] for i in pick])
        draws.append(metrics(y[ix], logit[ix], p_serv[ix], thr, full=False))
    return {k: [round(float(np.percentile([d[k] for d in draws], q)), 4) for q in (2.5, 97.5)] for k in draws[0]}


def per_speaker_fa(t, sel, p, thr) -> dict:
    """Calls cluster by speaker, so the call bootstrap understates speaker variance; report the spread."""
    spk = t["call_speaker"][t["call_index"][sel]]
    g = t["y"][sel] == 0
    return {s: round(float((p[g & (spk == s)] >= thr).mean()), 4) for s in np.unique(spk[g])}


def call_level(t, sel_split, p_serv, thr) -> dict:
    """Illustrative only (the backend owns the call policy): any-window and 3-consecutive-hop alarms."""
    out = {}
    for kind in ("genuine", "synthetic"):
        hits_any, hits3 = [], []
        for c in np.flatnonzero((t["call_kind"] == kind) & (t["call_split"] == sel_split)):
            a = p_serv[t["call_index"] == c] >= thr
            hits_any.append(a.any())
            hits3.append(bool(np.convolve(a, np.ones(3), "valid").max(initial=0) >= 3))
        key = "genuine_calls_alarmed" if kind == "genuine" else "synthetic_calls_detected"
        out[key] = {"n": len(hits_any), "any_window": round(float(np.mean(hits_any)), 4),
                    "3_consecutive_windows": round(float(np.mean(hits3)), 4)}
    return out


def rnd(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def benchmark(d: Path, rng) -> dict:
    out = {}
    for ch in present(d):
        t = load(d, ch)
        cal = t["meta"]["calibration"]
        sel = (t["win_split"] == "eval") & (t["y"] >= 0)
        y, lg = t["y"][sel].astype(int), t["logit"][sel].astype(np.float64)
        p = sig(cal["a"] * lg + cal["b"])
        thr = cal["decision_threshold"]
        out[ch] = {"n_windows": int(sel.sum()), "n_genuine_windows": int((y == 0).sum()),
                   "n_calls": int(len(np.unique(t["call_index"][sel]))),
                   **rnd(metrics(y, lg, p, thr)), "ci95": bootstrap(t, sel, y, lg, p, thr, rng),
                   "genuine_FA_by_speaker": per_speaker_fa(t, sel, p, thr),
                   "call_level_illustrative": call_level(t, "eval", sig(cal["a"] * t["logit"] + cal["b"]), thr)}
        print(Path(d).name, ch, {k: out[ch][k] for k in ("EER", "genuine_FA_at_serving_thr", "minDCF")}, flush=True)
    return out


def ece_balanced(y, p, n=10) -> float:
    w = np.where(y == 1, 0.5 / (y == 1).sum(), 0.5 / (y == 0).sum())
    b = np.clip((p * n).astype(int), 0, n - 1)
    return float(sum(abs(np.sum(w[b == k] * (y[b == k] - p[b == k]))) for k in range(n)))


def fit_platt(lg, y) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression

    w = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(lg[:, None], y, sample_weight=w)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def fit_temperature(z, y) -> float:
    from scipy.optimize import minimize_scalar

    w = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))

    def nll(log_t):
        p = np.clip(sig(z / np.exp(log_t)), 1e-7, 1 - 1e-7)
        return -np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p)))

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def calibration(d: Path) -> tuple[dict, dict]:
    tr = {ch: load(d, ch) for ch in CHANNELS}
    serv = tr["clean"]["meta"]["calibration"]
    fams, report = {}, {}
    for fam in ("clean", "telephone"):
        chs = [c for c in CHANNELS if FAMILY[c] == fam and c != "replay_sim"]  # replay_sim: held out, uses clean
        lg, y = [], []
        for c in chs:
            t = tr[c]
            s = (t["win_split"] == "calib") & (t["y"] >= 0) & (t["speech_s"] >= MIN_SPEECH_FIT)
            lg.append(t["logit"][s]), y.append(t["y"][s])
        lg, y = np.concatenate(lg).astype(np.float64), np.concatenate(y).astype(int)
        a, b = fit_platt(lg, y)
        p = sig(a * lg + b)
        md, thr = fast_mindcf(y, p)
        temp = fit_temperature(serv["a"] * lg + serv["b"], y)
        fams[fam] = {"a": a, "b": b, "decision_threshold": thr,
                     "threshold_window_FA_1pct": float(np.quantile(p[y == 0], 0.99)),
                     "temperature_on_serving_logit": temp, "calib_minDCF": md,
                     "fitted_channels": chs, "n_calib_windows": int(len(y))}
    for ch in CHANNELS:
        t, f = tr[ch], fams[FAMILY[ch]]
        s = (t["win_split"] == "eval") & (t["y"] >= 0)
        y, lg = t["y"][s].astype(int), t["logit"][s].astype(np.float64)
        zs = serv["a"] * lg + serv["b"]
        p_serv, p_temp, p_chan = sig(zs), sig(zs / f["temperature_on_serving_logit"]), sig(f["a"] * lg + f["b"])
        g = t["speech_s"][s] >= MIN_SPEECH_FIT
        row = {"family": FAMILY[ch]}
        for name, p, thr in (("serving", p_serv, serv["decision_threshold"]),
                             ("temperature", p_temp, serv["decision_threshold"]),
                             ("channel_platt", p_chan, f["decision_threshold"]),
                             ("channel_platt_fa1pct_thr", p_chan, f["threshold_window_FA_1pct"])):
            pred = p >= thr
            fa, miss = float(pred[y == 0].mean()), float((~pred[y == 1]).mean())
            row[name] = rnd({"ECE_balanced_10bin": ece_balanced(y, p), "ECE_balanced_speech_ge_1s": ece_balanced(y[g], p[g]),
                             "genuine_FA": fa, "spoof_miss": miss, "actDCF": fa + 4 * miss, "threshold": float(thr)})
        report[ch] = row
    return fams, report


def apply_gate(t, var, op, v):
    return t[var] >= v if op == ">=" else t[var] <= v


def abstention(d: Path, split: str) -> dict:
    tr = {ch: load(d, ch) for ch in present(d)}
    views = {ch: [ch] for ch in tr} | {"telephone_family": [c for c in tr if FAMILY[c] == "telephone"]}
    out = {}
    for view, chs in views.items():
        out[view] = {}
        for var, (op, vals) in GATES.items():
            rows = []
            for v in vals:
                ys, lgs, ps, cov_n, tot, gen_tot = [], [], [], 0, 0, 0
                for c in chs:
                    t = tr[c]
                    cal = t["meta"]["calibration"]
                    base = (t["win_split"] == split) & (t["y"] >= 0)
                    s = base & apply_gate(t, var, op, v)
                    tot += base.sum()
                    gen_tot += (base & (t["y"] == 0)).sum()
                    cov_n += s.sum()
                    ys.append(t["y"][s]), lgs.append(t["logit"][s]), ps.append(sig(cal["a"] * t["logit"][s] + cal["b"]))
                y, lg, p = np.concatenate(ys).astype(int), np.concatenate(lgs), np.concatenate(ps)
                thr = tr[chs[0]]["meta"]["calibration"]["decision_threshold"]
                row = {"gate": f"{var}{op}{v if v > -1e8 else '-inf'}", "coverage": round(cov_n / tot, 4)}
                if len(np.unique(y)) == 2 and min((y == 0).sum(), (y == 1).sum()) >= 20:
                    m = metrics(y, lg, p, thr, full=False)
                    row |= rnd(m | {"balanced_error": (m["genuine_FA_at_serving_thr"] + m["spoof_miss_at_serving_thr"]) / 2,
                                    "genuine_coverage": float((y == 0).sum() / gen_tot)})
                rows.append(row)
            out[view][var] = rows
    return out


def gated(d: Path, gate: list[tuple[str, str, float]]) -> dict:
    out = {}
    for ch in present(d):
        t = load(d, ch)
        cal = t["meta"]["calibration"]
        base = (t["win_split"] == "eval") & (t["y"] >= 0)
        s = base.copy()
        for var, op, v in gate:
            s &= apply_gate(t, var, op, v)
        y, lg = t["y"][s].astype(int), t["logit"][s].astype(np.float64)
        out[ch] = {"coverage": round(float(s.sum() / base.sum()), 4),
                   **rnd(metrics(y, lg, sig(cal["a"] * lg + cal["b"]), cal["decision_threshold"]))}
    return out


def parse_gate(s: str) -> list[tuple[str, str, float]]:
    out = []
    for part in filter(None, s.split(",")):
        op = ">=" if ">=" in part else "<="
        var, v = part.split(op)
        out.append((var.strip(), op, float(v)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", default=["outputs/telephony_eval"])
    ap.add_argument("--output", default="outputs/results/telephony_eval.json")
    ap.add_argument("--write-calibration", action="store_true", help="write models/selected/channel_calibration.json")
    ap.add_argument("--gate", default="speech_s>=1.0,snr_db>=20,loss_frac<=0.1",
                    help="combined quality gate reported as 'gated' (choose it on calib, see README)")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    res = {"protocol": {"calls": "outputs/telephony_eval/calls.json", "split_reported": "eval (speaker-disjoint from calib)",
                        "window_s": 3.0, "hop_s": 0.5, "n_bootstrap": N_BOOT, "bootstrap_unit": "call, stratified by kind",
                        "minDCF_costs": "Pspoof 0.5, Cmiss 1, Cfa 4 (HEARSAY)", "gate": a.gate,
                        "note": "Simulated calls and channels from the validation set; not real telephone data."},
           "models": {}, "abstention": {}}
    gate = parse_gate(a.gate)
    for d in a.traces:
        d = ROOT / d
        meta = load(d, "clean")["meta"]
        name = Path(meta["model_dir"]).name + ("_nofloor_ablation" if meta.get("no_floor") else "")
        res["models"][name] = {"traces": str(d.relative_to(ROOT)), "model_version": meta["model_version"],
                               "all_windows": benchmark(d, rng),
                               "gated": gated(d, gate)}
        splits = np.unique(load(d, "clean")["call_split"])
        res["abstention"][name] = {sp: abstention(d, sp) for sp in splits}
    first = ROOT / a.traces[0]
    if "calib" in np.unique(load(first, "clean")["call_split"]):
        fams, rep = calibration(first)
        res["channel_calibration"] = {"model": load(first, "clean")["meta"]["model_version"], "families": fams,
                                      "eval": rep}
        if a.write_calibration:
            cc = {fam: {"a": f["a"], "b": f["b"], "decision_threshold": f["decision_threshold"],
                        "threshold_window_FA_1pct": f["threshold_window_FA_1pct"]} for fam, f in fams.items()}
            cc["fitted_on"] = (f"outputs/telephony_eval calib split (speaker-disjoint), 3 s windows / 0.5 s hop, "
                               f"speech_s >= {MIN_SPEECH_FIT}, class-balanced Platt; decision_threshold = minDCF "
                               f"(Pspoof 0.5, Cfa 4) on calib. clean: {fams['clean']['fitted_channels']}; "
                               f"telephone: {fams['telephone']['fitted_channels']}. Simulated channels.")
            cc["convention"] = "p = sigmoid(a * logit + b), same as calibration.json"
            cc["eval"] = {ch: {k: r[k] for k in ("family", "serving", "channel_platt")} for ch, r in rep.items()}
            save_json(cc, ROOT / "models/selected/channel_calibration.json")
            print("wrote models/selected/channel_calibration.json")
    save_json(res, ROOT / a.output)
    print("wrote", a.output)


if __name__ == "__main__":
    main()
