"""Guardian HTTP/WebSocket routes (live dashboard feed, simulated calls, verification outcomes) and engine routes."""
from __future__ import annotations

import asyncio
import json
import random
import uuid
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app import config
from app.runtime import RT
from app.telephony import sim
from app.verification.alerts import dev_mode, user_from_token

guardian = APIRouter(prefix="/guardian", tags=["guardian"])
engine = APIRouter(tags=["engine"])
_bg: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    t = asyncio.get_running_loop().create_task(coro)
    _bg.add(t)
    t.add_done_callback(_bg.discard)


async def require_user(authorization: str | None) -> str:
    token = authorization.split(" ", 1)[1] if authorization and authorization.lower().startswith("bearer ") else None
    uid = await user_from_token(token)
    if not uid:
        raise HTTPException(401, "sign in required")
    return uid


@guardian.websocket("/live")
async def live(ws: WebSocket, token: str | None = None):
    await ws.accept()
    uid = await user_from_token(token)
    if not uid:
        await ws.send_json({"type": "error", "detail": "sign in required"})
        await ws.close(code=1008)
        return
    q = RT.hub.subscribe(uid)
    await ws.send_json({"type": "hello", "user_id": uid, "dev_mode": dev_mode()})
    for s in list(RT.calls.values()):  # late joiners get the current state of their active calls
        if s.user_id == uid:
            await ws.send_json(s.started_ev)
            if s.last_update:
                await ws.send_json(s.last_update)
    try:
        while True:
            await ws.send_json(await q.get())
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        RT.hub.unsubscribe(uid, q)


class SimulateReq(BaseModel):
    scenario: Literal["genuine", "synthetic", "mixed"] = "synthetic"
    transport: Literal["in_process", "external"] = "in_process"


@guardian.post("/simulate")
async def simulate(req: SimulateReq, authorization: str | None = Header(None)):
    """Simulated call for demos/dev. `external` returns a stream token for `python -m app.telephony.sim`."""
    uid = await require_user(authorization)
    sid = str(uuid.uuid4())
    spec = {"user_id": uid, "caller": "simulated caller", "consent": "simulated", "scenario": req.scenario,
            "session_id": sid}
    if req.transport == "external":
        token = uuid.uuid4().hex
        RT.pending[token] = spec
        return {"session_id": sid, "token": token}
    try:
        audio = await asyncio.to_thread(sim.scenario_audio, req.scenario)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    _spawn(sim.run_in_process(audio, spec))
    return {"session_id": sid}


class VerificationReq(BaseModel):
    contact_id: str | None = None
    outcome: Literal["verified_genuine", "could_not_reach", "confirmed_scam", "skipped"]


@guardian.post("/sessions/{session_id}/verification")
async def verification(session_id: str, req: VerificationReq, authorization: str | None = Header(None)):
    uid = await require_user(authorization)
    live_s = RT.calls.get(session_id)
    owner = live_s.user_id if live_s else (await RT.store.session(session_id) or {}).get("user_id")
    if owner != uid:
        raise HTTPException(404, "session not found")
    RT.store.add_event(session_id, uid, "verification", detail=req.model_dump())
    return {"ok": True}


# --- engine ---------------------------------------------------------------------------------------------------
@engine.get("/engine/metrics")
def metrics():
    return RT.metrics()


@engine.get("/metrics", response_class=PlainTextResponse)
def prometheus():
    return RT.telemetry.prometheus(RT.metrics())


class PolicyReq(BaseModel):
    policy: Literal["fifo", "dynamic", "edf", "adaptive"]


@engine.post("/engine/policy")
def set_policy(req: PolicyReq):
    RT.aegis.set_policy(req.policy)
    RT.telemetry.reset()
    return RT.metrics()


class LoadReq(BaseModel):
    calls: int = Field(10, ge=1, le=64)
    seconds: float = Field(30, ge=5, le=300)
    scenario: Literal["genuine", "synthetic", "mixed"] = "mixed"


@engine.post("/engine/loadtest")
async def loadtest(req: LoadReq):
    """Concurrent simulated callers (no owner: visible only in engine metrics). Staggered starts."""
    try:
        audios = await asyncio.to_thread(
            lambda: [sim.scenario_audio(random.choice(["genuine", "synthetic"]) if req.scenario == "mixed"
                                        else req.scenario, seed=i, target_s=req.seconds) for i in range(req.calls)])
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e

    async def one(i, a):
        await asyncio.sleep(random.uniform(0, 0.5))
        await sim.run_in_process(a[: int(req.seconds * 16000)], {"user_id": None, "caller": f"load-{i}",
                                                                  "consent": "simulated", "scenario": "load"})

    for i, a in enumerate(audios):
        _spawn(one(i, a))
    return {"started": req.calls}


@engine.get("/engine/benchmarks")
def benchmarks():
    d = config.OUTPUT_DIR / "benchmarks"
    reps = []
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        data = json.loads(p.read_text())
        reps.append({"name": p.stem, "created": data.get("created"), "data": data})
    return {"reports": reps}
