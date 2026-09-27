#!/usr/bin/env python3
"""Aggregate measured results into outputs/eval_summary.json (served by GET /eval/summary) and RESULTS.md.
Only reads result files produced by the ML scripts; nothing is typed in by hand. Missing results are omitted.

    python ml/build_eval_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
from common import manifest  # noqa: E402

R = ROOT / "outputs/results"


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def r4(x):
    return None if x is None else round(float(x), 4)


def main():
    S: dict = {"notes": [
        "minDCF = official organizer scorer (ASVspoof5 Track 1 code, Pspoof=0.5, Cmiss=1, Cfa=4); lower is better.",
        "Validation = group-disjoint split of organizer DiffSSD + LJRealResampled + public LibriSpeech dev/test-clean "
        "bona fide, scored on test-like 3.0-4.6 s crops.",
        "Multi-speaker spoofs in validation are from 2 cloned speakers never seen in training.",
        "Unseen-generator rows: generator removed from training entirely.",
    ]}
    sel_dir = ROOT / "models/selected"
    sel = load(sel_dir / "val_report_serving.json")
    rows = []
    zs = load(R / "baseline_aasist_zeroshot.json")
    if zs:
        rows.append({"model": "AASIST (organizer ASVspoof5 ckpt), zero-shot", "val_minDCF": r4(zs["pooled"]["minDCF"]),
                     "EER": r4(zs["pooled"]["EER"]), "AUC": r4(zs["pooled"]["AUC"])})
    fus = load(R / "fusion.json")
    if fus:
        s = fus["standalone_train_to_val"]["all_branches"]
        rows.append({"model": "Hand-crafted forensic branches (38 feats, logistic regression)",
                     "val_minDCF": s["minDCF"], "EER": s["EER"], "AUC": s["AUC"]})
    for d in sorted((ROOT / "models").glob("*/val_report*.json")):
        if d.parent.name in ("selected", "hf_cache"):
            continue
        rep = load(d)
        if d.name == "val_report.json" and (d.parent / "val_report_serving.json").exists():
            continue
        rows.append({"model": d.parent.name, "val_minDCF": r4(rep["pooled"]["minDCF"]), "EER": r4(rep["pooled"]["EER"]),
                     "AUC": r4(rep["pooled"]["AUC"]),
                     "LJ_bona_vs_LJ_voice_spoofs_minDCF": rep.get("lj_bonafide_vs_lj_voice_spoofs_minDCF"),
                     "Libri_bona_vs_cloned_spoofs_minDCF": rep.get("libri_bonafide_vs_cloned_spoofs_minDCF")})
    S["model_selection"] = rows
    if sel:
        meta = load(sel_dir / "meta.json")
        S["selected_model"] = {"name": meta.get("name"), "backbone": meta.get("backbone"),
                               "val_minDCF": r4(sel["pooled"]["minDCF"]), "EER": r4(sel["pooled"]["EER"]),
                               "AUC": r4(sel["pooled"]["AUC"]), "actDCF": r4(sel["pooled"]["actDCF"]),
                               "decision_threshold": r4(sel["calibration"]["decision_threshold"]),
                               "FRR_bonafide_at_minDCF": r4(sel["pooled"]["FRR_bonafide_at_minDCF"]),
                               "FAR_spoof_at_minDCF": r4(sel["pooled"]["FAR_spoof_at_minDCF"])}
        S["per_generator_minDCF"] = [{"generator": g, "minDCF": v} for g, v in sel["per_generator_minDCF"].items()]
        lg = np.load(sel_dir / "val_logits_serving.npy")
        cal = sel["calibration"]
        p = 1 / (1 + np.exp(-(cal["a"] * lg + cal["b"])))
        y = manifest("val").y.to_numpy()
        bins = np.linspace(0, 1, 21)
        S["histograms"] = {"validation_synthetic_probability": {
            "bins": [round(b, 2) for b in bins[:-1]],
            "bonafide": np.histogram(p[y == 0], bins)[0].tolist(), "spoof": np.histogram(p[y == 1], bins)[0].tolist()}}
    uns = []
    for f in sorted(R.glob("unseen_*.json")):
        u = load(f)
        uns.extend(u["rows"])
    if uns:
        S["unseen_generator"] = uns
    if fus:
        S["forensic_branches_standalone"] = [{"branch": k, **v} for k, v in fus["standalone_train_to_val"].items()]
        S["fusion_ablation_oof"] = [{"config": k, "minDCF": v} for k, v in fus["fusion_oof_on_val"].items()]
        S["fusion_decision"] = fus["decision"]
    rob = load(R / "robustness.json")
    if rob:
        S["robustness"] = [{"condition": k, **v} for k, v in rob["conditions"].items()]
    lat = load(R / "latency.json")
    if lat:
        S["latency"] = lat
    agg = load(R / "aggregation.json")
    if agg:
        S["clip_aggregation"] = agg["rows"]
    ood = load(R / "grok_ood.json")
    if ood:
        S["grok_voice_ood"] = ood["rows"]
    held = load(ROOT / "outputs/team_predictions.json")
    if held:
        S["heldout_prediction_stats"] = held["score_stats"] | {"model_sha1": held["model_sha1"]}
    (ROOT / "outputs/eval_summary.json").write_text(json.dumps(S, indent=2), encoding="utf-8")
    write_results_md(S)
    print("wrote outputs/eval_summary.json and RESULTS.md")


def table(rows: list[dict]) -> str:
    if not rows:
        return ""
    cols = list(dict.fromkeys(k for r in rows for k in r))
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def write_results_md(S: dict):
    parts = ["# RESULTS.md — ProofVoice (auto-generated by `ml/build_eval_summary.py`; do not edit by hand)\n",
             "\n".join(f"- {n}" for n in S["notes"]), ""]
    if "selected_model" in S:
        parts += ["## Selected model", table([S["selected_model"]]), ""]
    for key, title in [("model_selection", "Model selection (validation)"),
                       ("per_generator_minDCF", "Per-generator minDCF (each generator vs all bona fide)"),
                       ("unseen_generator", "Unseen-generator generalisation"),
                       ("forensic_branches_standalone", "Forensic branches — standalone (train → val)"),
                       ("fusion_ablation_oof", "Fusion ablation — out-of-fold on validation"),
                       ("robustness", "Robustness / laundering"),
                       ("grok_voice_ood", "Out-of-distribution: live Grok Voice samples"),
                       ("clip_aggregation", "Long-clip aggregation (full-length val subsample, n=2000)")]:
        if key in S:
            parts += [f"## {title}", table(S[key]), ""]
    if "fusion_decision" in S:
        parts += [f"**Fusion decision:** {S['fusion_decision']}", ""]
    if "latency" in S:
        parts += ["## Latency", "```json", json.dumps(S["latency"], indent=1), "```", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
