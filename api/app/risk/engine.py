"""Call-level risk policy: sequential evidence accumulation with abstention (PRD §6 Workstream B).

Per window (3 s, 0.5 s hop) the detector gives a raw logit. It is turned into threshold-centred log-odds with the
channel-conditioned calibration (p = sigmoid(a·logit + b); llr = (a·logit + b) − logit(threshold)), clipped, and
weighted by signal quality. Unusable windows (too little speech, low SNR, heavy packet loss) contribute nothing.

    S_t = max(0, S_{t-1} + w_t · ρ · llr_t)        one-sided CUSUM toward "synthetic"
    ρ   = hop / window                             overlapping windows share audio: count each hop once

`suspicious`  ⇐ S ≥ H and ≥ `persist` of the last 6 usable windows above threshold (latched for the call)
`no_suspicious_evidence` ⇐ ≥ clear_speech_s of usable speech, S < H/4 and ≤ 20 % of recent usable windows above threshold
`inconclusive` otherwise (with the reason), `analysis_unavailable` when set by the transport/scheduler.

A low score never establishes identity; the UI copy says so. Parameters come from `risk_policy.json` next to the
model when present (fitted by benchmarks/policy_replay.py on calibration calls), else the defaults below.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

STATES = ("no_suspicious_evidence", "inconclusive", "suspicious", "analysis_unavailable")


@dataclass
class RiskPolicy:
    H: float = 3.0
    max_llr: float = 4.0
    min_speech_s: float = 1.0
    min_snr_db: float = 12.0
    full_snr_db: float = 30.0
    max_packet_loss: float = 0.10
    clear_speech_s: float = 6.0
    persist: int = 3
    rho: float = 0.5 / 3.0

    @classmethod
    def load(cls, model_dir: str | Path | None) -> "RiskPolicy":
        p = Path(model_dir) / "risk_policy.json" if model_dir else None
        if p and p.exists():
            d = json.loads(p.read_text())
            return cls(**{f.name: d[f.name] for f in fields(cls) if f.name in d})
        return cls()


@dataclass
class Calibration:
    a: float = 1.0
    b: float = 0.0
    threshold: float = 0.5

    @classmethod
    def load(cls, model_dir: str | Path, channel: str) -> "Calibration":
        """Channel-conditioned calibration if fitted (models/*/channel_calibration.json), else the global one."""
        d = Path(model_dir)
        for name, key in (("channel_calibration.json", channel), ("calibration.json", None)):
            p = d / name
            if p.exists():
                c = json.loads(p.read_text())
                c = c.get(key) if key else c
                if c:
                    return cls(float(c["a"]), float(c["b"]), float(c["decision_threshold"]))
        return cls()

    def log_odds(self, logit: float) -> float:
        return self.a * logit + self.b

    def prob(self, logit: float) -> float:
        return 1 / (1 + math.exp(-max(-60.0, min(60.0, self.log_odds(logit)))))


class RiskEngine:
    def __init__(self, calibration: Calibration, policy: RiskPolicy | None = None):
        self.cal, self.pol = calibration, policy or RiskPolicy()
        self.thr_lo = math.log(self.cal.threshold / (1 - self.cal.threshold))
        self.S = 0.0
        self.peak = 0.0
        self.usable_speech_s = 0.0
        self.recent: list[bool] = []      # usable windows: above threshold?
        self.recent_llr: list[float] = []  # inferred windows (for the scheduler gate)
        self.state = "inconclusive"
        self.reason = "Listening for speech."
        self.unavailable: str | None = None

    @property
    def evidence(self) -> float:
        return round(min(1.5, self.S / self.pol.H), 3)

    def weight(self, q: dict) -> float:
        p = self.pol
        if q.get("speech_s", 0) < p.min_speech_s or q.get("snr_db", 0) < p.min_snr_db \
                or q.get("packet_loss", 0) > p.max_packet_loss:
            return 0.0
        w_snr = min(1.0, (q["snr_db"] - p.min_snr_db) / (p.full_snr_db - p.min_snr_db))
        return round(max(0.0, w_snr) * (1 - q.get("packet_loss", 0) / p.max_packet_loss * 0.5), 3)

    def update(self, logit: float, q: dict, hop_s: float = 0.5) -> dict:
        """One inferred window. Returns the window record for the UI/timeline."""
        llr = max(-self.pol.max_llr, min(self.pol.max_llr, self.cal.log_odds(logit) - self.thr_lo))
        self.recent_llr = (self.recent_llr + [llr])[-12:]
        w = self.weight(q)
        if w > 0:
            self.S = max(0.0, self.S + w * self.pol.rho * llr)
            self.peak = max(self.peak, self.S)
            self.usable_speech_s += min(hop_s, q["speech_s"])
            self.recent = (self.recent + [llr > 0])[-12:]
        self._settle(q, w)
        return {"synthetic_probability": round(self.cal.prob(logit), 4), "usable": w > 0, "weight": w,
                "llr": round(llr, 3)}

    def set_unavailable(self, why: str | None) -> None:
        self.unavailable = why
        if why:
            self.state, self.reason = "analysis_unavailable", why
        elif self.state == "analysis_unavailable":
            self.state, self.reason = "inconclusive", "Analysis resumed."

    def _settle(self, q: dict, w: float) -> None:
        p = self.pol
        if self.unavailable:
            return
        if self.state == "suspicious":  # latched: a warning must not flicker away mid-call
            return
        last6 = self.recent[-6:]
        if self.S >= p.H and sum(last6) >= p.persist:
            self.state = "suspicious"
            self.reason = ("Persistent synthetic-speech patterns across multiple stretches of speech. "
                           "Verify independently before sharing information or sending money.")
            return
        frac = sum(self.recent) / len(self.recent) if self.recent else 0.0
        if self.usable_speech_s >= p.clear_speech_s and self.S < p.H / 4 and frac <= 0.2:
            self.state = "no_suspicious_evidence"
            self.reason = ("No strong synthetic-speech signal in the speech so far. This does not prove who "
                           "the caller is.")
            return
        self.state = "inconclusive"
        if w == 0 and q.get("speech_s", 0) >= p.min_speech_s and q.get("snr_db", 0) < p.min_snr_db:
            self.reason = f"Audio too noisy to judge reliably (≈{q['snr_db']:.0f} dB SNR)."
        elif w == 0 and q.get("packet_loss", 0) > p.max_packet_loss:
            self.reason = f"Too much of the call audio is missing ({q['packet_loss']:.0%} lost)."
        elif self.usable_speech_s < p.clear_speech_s and frac <= 0.2:
            self.reason = f"Gathering clear speech ({self.usable_speech_s:.1f} of {p.clear_speech_s:.0f} s)."
        else:
            self.reason = "Mixed signals: some stretches of speech look synthetic. Still gathering evidence."

    def summary(self) -> dict:
        return {"state": self.state, "evidence": self.evidence, "peak_evidence": round(min(1.5, self.peak / self.pol.H), 3),
                "usable_speech_s": round(self.usable_speech_s, 2), "policy": asdict(self.pol)}


if __name__ == "__main__":
    good = {"speech_s": 2.5, "snr_db": 35.0, "packet_loss": 0.0}
    cal = Calibration(1.0, 0.0, 0.5)
    r = RiskEngine(cal)
    for _ in range(40):  # genuine-looking: llr = -3
        r.update(-3.0, good)
    assert r.state == "no_suspicious_evidence" and r.S == 0.0, r.summary()
    r = RiskEngine(cal)
    n = 0
    while r.state != "suspicious":
        r.update(3.0, good)
        n += 1
    assert 5 <= n <= 8, n  # H=3, ρ=1/6, llr=3 -> 6 windows ≈ 3 s of hops
    r.update(-4.0, good)
    assert r.state == "suspicious"  # latched
    r = RiskEngine(cal)
    for _ in range(40):  # noisy: never usable -> abstain
        r.update(3.0, {"speech_s": 2.5, "snr_db": 8.0, "packet_loss": 0.0})
    assert r.state == "inconclusive" and "noisy" in r.reason, r.reason
    r.set_unavailable("Analysis paused: server overloaded.")
    assert r.state == "analysis_unavailable"
    print("risk engine ok")
