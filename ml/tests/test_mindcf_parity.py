"""minDCF wrapper parity + regression tests against the organizer package (HackGTMinDCF.zip)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
import evaluate_mindcf as em  # noqa: E402

FIX = ROOT / "ml/official_scoring"


def test_official_fixture_regression():
    """Organizer example (test_cm_file / test_cm_key) -> their shipped track1_result.txt: minDCF 0.857142857..., EER 58.571%."""
    scores, labels = em._load(str(FIX / "test_cm_file"), str(FIX / "test_cm_key"))
    r = em.official_metrics(scores[labels == "bonafide"], scores[labels == "spoof"])
    assert r["minDCF"] == pytest.approx(0.8571428571428571, abs=1e-12)
    assert r["EER"] * 100 == pytest.approx(58.571428571, abs=1e-6)
    assert r["CLLR"] == pytest.approx(0.975675585, abs=1e-8)


def test_cli_matches_organizer_script(tmp_path):
    """Run the organizer evaluation.py as a subprocess and compare with our wrapper on a random fixture."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.random(n) < 0.3
    s = np.where(y, rng.normal(-1, 1, n), rng.normal(1, 1, n))
    names = [f"F{i:05d}" for i in range(n)]
    (tmp_path / "s.tsv").write_text("filename\tcm-score\n" + "".join(f"{a}\t{b:.6f}\n" for a, b in zip(names, s)))
    (tmp_path / "k.tsv").write_text("filename\tcm-label\n" + "".join(
        f"{a}\t{'spoof' if t else 'bonafide'}\n" for a, t in zip(names, y)))
    out = subprocess.run([sys.executable, str(FIX / "evaluation.py"), "--m", "t1", "--cm", str(tmp_path / "s.tsv"),
                          "--cm_keys", str(tmp_path / "k.tsv")], cwd=tmp_path, capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if l.startswith("-eval_mindcf")][0]
    official = float(line.split(":")[1])
    sc, lb = em._load(str(tmp_path / "s.tsv"), str(tmp_path / "k.tsv"))
    ours = em.official_metrics(sc[lb == "bonafide"], sc[lb == "spoof"])["minDCF"]
    assert ours == pytest.approx(official, abs=1e-5)


def test_fast_equals_official_and_orientation():
    rng = np.random.default_rng(1)
    y = rng.random(2000) < 0.5
    p = np.clip(np.where(y, rng.beta(5, 2, 2000), rng.beta(2, 5, 2000)), 0, 1)
    fast, thr = em.fast_mindcf(y, p)
    assert fast == pytest.approx(em.mindcf_from_psynth(y, p)["minDCF"], abs=1e-12)
    assert 0 <= thr <= 1
    # inverted orientation must be much worse (guards against sign bugs)
    assert em.mindcf_from_psynth(y, 1 - p)["minDCF"] > fast + 0.3


def test_cost_model_is_organizer_modified():
    assert (em.PSPOOF, em.CMISS, em.CFA) == (0.5, 1.0, 4.0)
    src = (FIX / "calculate_metrics.py").read_text()
    assert "Pspoof = 0.5" in src and "'Cfa' : 4" in src


def test_perfect_and_trivial():
    y = np.array([0] * 70 + [1] * 30)
    assert em.fast_mindcf(y, y.astype(float))[0] == pytest.approx(0.0)
    assert em.fast_mindcf(y, np.full(100, 0.5))[0] == pytest.approx(1.0)
