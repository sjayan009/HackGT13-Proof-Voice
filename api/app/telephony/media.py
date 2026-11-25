"""Live call analysis: Twilio Media Stream frames -> 16 kHz ring buffer -> Aegis gate/inference -> risk -> events.

Audio lives only in a bounded in-memory ring buffer (8 s) and is discarded when the call ends.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time

import numpy as np
import soxr

from app.audio.realtime import FloorTracker, window_quality
from app.risk.engine import RiskEngine
from app.runtime import RT
from app.telephony.mulaw import decode
from app.verification.alerts import ALERT_MESSAGE, send_sms
from app.verification.store import now_iso

log = logging.getLogger("proofvoice.call")
SR, IN_SR = 16000, 8000
HOP, WIN = SR // 2, 3 * SR
BUF_MAX = 8 * SR
MAX_GAP_MS = 2000
OVERLOAD_SHEDS = 6  # consecutive shed hops (3 s without a result) -> analysis_unavailable


class CallSession:
    def __init__(self, *, user_id: str | None, call_sid: str, caller: str | None, consent: str,
                 line: dict | None = None, scenario: str | None = None, session_id: str | None = None):
        self.user_id, self.call_sid, self.caller, self.consent = user_id, call_sid, caller, consent
        self.line, self.scenario = line, scenario
        self.id = RT.store.create_session({"id": session_id, "user_id": user_id, "call_sid": call_sid,
                                           "caller_number": caller, "consent": consent,
                                           "line_id": (line or {}).get("id")})
        self.risk = RiskEngine(RT.calibration["telephone"], RT.risk_policy)
        self.rs = soxr.ResampleStream(IN_SR, SR, 1, dtype="float32", quality="HQ")
        self.buf = np.zeros(0, np.float32)
        self.lost = np.zeros(0, np.float32)
        self.total = 0
        self.next_at = WIN
        self.expected_ts: float | None = None
        self.floor = FloorTracker()
        self.hops_since_infer = 0
        self.sheds = 0
        self.inferred = self.skipped = 0
        self.alerted = False
        self.t0 = time.perf_counter()
        self._new = asyncio.Event()
        self._ended = False
        RT.calls[self.id] = self
        self.started_ev = {"type": "call.started", "session_id": self.id, "call_sid": call_sid, "caller": caller,
                           "consent": consent, "started_at": now_iso(), "scenario": scenario}
        self.last_update: dict | None = None  # replayed to dashboards that connect mid-call
        self._emit(self.started_ev)
        if consent == "declined":
            self.risk.set_unavailable("The caller declined analysis. The call is connected but not being checked.")
            self._emit(self._update(None, {"policy": RT.aegis.policy, "action": "shed", "latency_ms": None}))
            RT.store.add_event(self.id, user_id, "unavailable", state=self.risk.state, detail={"why": "consent_declined"})
            self._worker = None
        else:
            self._worker = asyncio.get_running_loop().create_task(self._run())

    # --- audio in ---------------------------------------------------------------------------------------------
    def _push(self, x8: np.ndarray, lost: bool) -> None:
        x = self.rs.resample_chunk(x8)
        self.total += len(x)
        self.buf = np.concatenate([self.buf, x])[-BUF_MAX:]
        self.lost = np.concatenate([self.lost, np.full(len(x), float(lost), np.float32)])[-BUF_MAX:]

    def feed(self, payload: bytes, timestamp_ms: float | None = None) -> None:
        """One Twilio media frame (μ-law 8 kHz). Gaps in Twilio's timestamps are zero-filled and counted as loss."""
        if self._ended or self._worker is None:
            return
        if timestamp_ms is not None and self.expected_ts is not None and timestamp_ms > self.expected_ts + 10:
            gap = min(MAX_GAP_MS, timestamp_ms - self.expected_ts)
            self._push(np.zeros(int(gap * IN_SR / 1000), np.float32), lost=True)
        self._push(decode(payload), lost=False)
        if timestamp_ms is not None:
            self.expected_ts = timestamp_ms + len(payload) * 1000 / IN_SR
        if self.total >= self.next_at:
            self._new.set()

    # --- analysis loop ----------------------------------------------------------------------------------------
    async def _run(self) -> None:
        """Owns the session lifecycle: drains pending hops after end() and always finalises (even on errors)."""
        try:
            while True:
                await self._new.wait()
                self._new.clear()
                while self.total >= self.next_at:
                    behind = (self.total - self.next_at) // HOP  # hops that arrived while we were busy
                    for _ in range(behind):
                        self._hop_skipped("shed")
                    self.next_at += (behind + 1) * HOP
                    await self._process_hop()
                if self._ended:
                    return
        finally:
            self._finalize()

    def _hop_skipped(self, action: str) -> None:
        self.skipped += 1
        RT.telemetry.hop(action)
        self._emit(self._update(None, {"policy": RT.aegis.policy, "action": action, "latency_ms": None}))

    async def _process_hop(self) -> None:
        win, hop = self.buf[-WIN:], self.buf[-HOP:]
        floor = self.floor.update(hop)
        q = window_quality(win, floor) | {"packet_loss": round(float(self.lost[-WIN:].mean()), 3)}
        hop_speech = window_quality(hop, floor)["speech_s"]
        prev = self.risk.state
        action = RT.aegis.decide(hop_speech_s=hop_speech, state=prev, recent_llr=self.risk.recent_llr,
                                 hops_since_infer=self.hops_since_infer)
        if action not in ("infer", "escalate"):  # skip_silence / skip_clear
            self.hops_since_infer += 1
            return self._hop_skipped(action)
        t = time.perf_counter()
        cal = None
        try:
            logit = None
            if RT.aegis.screening and action == "infer":  # two-stage: distilled student first
                from engine.aegis import SCREEN_CLEAR_LLR

                s_logit = await RT.aegis.infer(self.id, win.copy(), model="screen")
                if s_logit is not None and RT.screen_calibration.llr(s_logit) < SCREEN_CLEAR_LLR:
                    logit, cal, action = s_logit, RT.screen_calibration, "screen_only"
            if logit is None:
                logit = await RT.aegis.infer(self.id, win.copy(), escalate=action == "escalate")
        except Exception as e:  # noqa: BLE001
            log.exception("inference failed")
            logit, action = None, "shed"
            self.risk.set_unavailable(f"Analysis unavailable: inference error ({type(e).__name__}). The call continues.")
        lat = round((time.perf_counter() - t) * 1000, 1)
        window = None
        if logit is None:
            action = "shed"
            self.sheds += 1
            self.skipped += 1
            if self.sheds >= OVERLOAD_SHEDS and self.risk.state != "analysis_unavailable":
                self.risk.set_unavailable("Analysis delayed: the server is overloaded. The call continues.")
        else:
            self.sheds = 0
            if self.risk.unavailable and "overloaded" in self.risk.unavailable:
                self.risk.set_unavailable(None)
            self.inferred += 1
            self.hops_since_infer = 0
            rec = self.risk.update(logit, q, cal=cal)
            end_ms = int(self.total * 1000 / SR)
            window = {"start_ms": max(0, end_ms - 3000), "end_ms": end_ms, **rec,
                      "speech_s": q["speech_s"], "snr_db": q["snr_db"], "level_dbfs": q["level_dbfs"],
                      "packet_loss": q["packet_loss"]}
            window.pop("llr")
        RT.telemetry.hop(action)
        self._emit(self._update(window, {"policy": RT.aegis.policy, "action": action, "latency_ms": lat}))
        if self.risk.state != prev:
            RT.store.add_event(self.id, self.user_id, "unavailable" if self.risk.state == "analysis_unavailable"
                               else "state", t_ms=self.t_ms, state=self.risk.state, evidence=self.risk.evidence,
                               detail={"reason": self.risk.reason})
            if self.risk.state == "suspicious" and not self.alerted:
                await self._alert()

    async def _alert(self) -> None:
        self.alerted = True
        sms = False
        recipient = (self.line or {}).get("recipient_phone")
        if recipient and self.consent == "granted":
            sms = await send_sms(recipient, f"ProofVoice Guardian: {ALERT_MESSAGE} Hang up and call the person "
                                            "back on a number you already know.")
        self._emit({"type": "call.alert", "session_id": self.id, "t_ms": self.t_ms, "evidence": self.risk.evidence,
                    "message": ALERT_MESSAGE, "sms_sent": sms})
        RT.store.add_event(self.id, self.user_id, "alert", t_ms=self.t_ms, state="suspicious",
                           evidence=self.risk.evidence, detail={"sms_sent": sms})

    # --- events -----------------------------------------------------------------------------------------------
    @property
    def t_ms(self) -> int:
        return int(self.total * 1000 / SR)

    def _update(self, window: dict | None, compute: dict) -> dict:
        return {"type": "call.update", "session_id": self.id, "t_ms": self.t_ms, "state": self.risk.state,
                "evidence": self.risk.evidence, "reason": self.risk.reason, "window": window, "compute": compute}

    def _emit(self, ev: dict) -> None:
        if ev["type"] == "call.update":
            self.last_update = ev
        RT.hub.publish(self.user_id, ev)

    async def end(self) -> None:
        if self._ended:
            return
        self._ended = True
        self._new.set()
        if self._worker:
            # shield: if the transport handler is cancelled (abrupt disconnect, shutdown), the worker still
            # drains and finalises instead of inheriting the cancellation
            await asyncio.shield(self._worker)
        else:
            self._finalize()

    def _finalize(self) -> None:
        if self.id not in RT.calls:
            return
        s = self.risk.summary()
        self._emit({"type": "call.ended", "session_id": self.id, "final_state": s["state"],
                    "peak_evidence": s["peak_evidence"], "duration_ms": self.t_ms,
                    "usable_speech_s": s["usable_speech_s"], "windows_inferred": self.inferred,
                    "windows_skipped": self.skipped})
        RT.store.end_session(self.id, self.user_id, {"final_state": s["state"], "peak_evidence": s["peak_evidence"],
                                                      "alerted": self.alerted})
        self.buf = self.lost = np.zeros(0, np.float32)  # drop audio immediately
        RT.calls.pop(self.id, None)


class TwilioStream:
    """Consumes Twilio Media Stream messages (dicts). The WebSocket route and the simulator share this code."""

    def __init__(self):
        self.session: CallSession | None = None

    async def handle(self, msg: dict) -> bool:
        """Returns False when the stream is finished."""
        ev = msg.get("event")
        if ev == "start":
            st = msg.get("start", {})
            spec = RT.pending.pop((st.get("customParameters") or {}).get("token", ""), None)
            if spec is None:
                raise PermissionError("unknown or reused stream token")
            fmt = st.get("mediaFormat") or {}
            if fmt and (fmt.get("encoding") != "audio/x-mulaw" or int(fmt.get("sampleRate", 8000)) != IN_SR):
                raise ValueError(f"unsupported media format {fmt}")
            self.session = CallSession(call_sid=st.get("callSid", ""), **spec)
        elif ev == "media" and self.session:
            m = msg["media"]
            if m.get("track", "inbound") == "inbound":
                self.session.feed(base64.b64decode(m["payload"]), float(m.get("timestamp", 0)))
        elif ev == "stop":
            await self.close()
            return False
        return True

    async def close(self) -> None:
        if self.session:
            await self.session.end()
