"""Process-wide v2 runtime (filled in main.lifespan): Aegis scheduler, telemetry, store, live hub, active calls."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Runtime:
    detector: object = None
    aegis: object = None
    telemetry: object = None
    store: object = None
    hub: object = None
    risk_policy: object = None
    calibration: dict = field(default_factory=dict)  # channel -> Calibration
    screen_calibration: object = None                # two-stage screening student (telephone channel)
    calls: dict = field(default_factory=dict)        # session_id -> CallSession
    pending: dict = field(default_factory=dict)      # stream token -> session spec (from /twilio/consent or simulate)

    def metrics(self) -> dict:
        tm, ag = self.telemetry, self.aegis
        return tm.snapshot(policy=ag.policy, backend=ag.backend.name, device=ag.backend.device_name,
                           active_calls=len(self.calls), queue_depth=ag.queue_depth, window_s=3.0, hop_s=0.5,
                           deadline_ms=ag.deadline * 1000)


RT = Runtime()
