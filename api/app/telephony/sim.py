"""Twilio Media Streams simulator: turns audio into the exact JSON frames Twilio sends (8 kHz μ-law, 20 ms).

In-process (`/guardian/simulate`, `/engine/loadtest`, benchmarks) the frames go through `TwilioStream.handle`; from
the CLI they go over a real WebSocket to `/twilio/media`:

    python -m app.telephony.sim ws://127.0.0.1:8000/twilio/media --scenario synthetic --token <token>

Scenario audio is drawn from local data (gitignored): genuine = LibriSpeech validation speakers; synthetic = the
Grok Voice scam-style samples (a generator never seen in training) or validation spoof clips; mixed = genuine
speech followed by synthetic speech. Simulated calls are always labelled as simulations.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import time
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np

from app import config
from app.audio.ingest import load_audio, resample
from app.telephony.mulaw import encode

FRAME = 160  # 20 ms at 8 kHz


@lru_cache(maxsize=1)
def _val_rows() -> list[dict]:
    p = config.OUTPUT_DIR / "manifests" / "val.csv"
    if not p.exists():
        return []
    with p.open() as f:
        return [r for r in csv.DictReader(f) if (config.ROOT / r["path"]).exists()]


def _concat(paths: list[Path], target_s: float, rng: np.random.Generator) -> np.ndarray:
    out, n = [], 0
    for p in paths:
        x, _, _ = load_audio(p)
        out += [x, np.zeros(int(rng.uniform(0.2, 0.7) * 16000), np.float32)]
        n += len(out[-2]) + len(out[-1])
        if n >= target_s * 16000:
            break
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def scenario_audio(scenario: str, seed: int | None = None, target_s: float = 28.0) -> np.ndarray:
    """16 kHz float32 audio for a simulated call. Raises FileNotFoundError if the local data is missing."""
    rng = np.random.default_rng(seed)
    rows = _val_rows()
    genuine = [r for r in rows if r["label"] == "bonafide" and r["source"] == "librispeech"]
    if scenario in ("genuine", "mixed") and not genuine:
        raise FileNotFoundError("genuine sample audio (data/external/LibriSpeech + val manifest) not available")
    grok = sorted((config.DATA_DIR / "grok_samples").glob("*.wav"))

    def genuine_audio(s):
        spk = rng.choice(sorted({r["speaker"] for r in genuine}))
        return _concat([config.ROOT / r["path"] for r in genuine if r["speaker"] == spk], s, rng)

    def synthetic_audio(s):
        if grok and (seed is None or rng.random() < 0.5):
            return _concat(list(rng.permutation(grok)) * 3, s, rng)
        spoof = [r for r in rows if r["label"] == "spoof"]
        if not spoof:
            raise FileNotFoundError("synthetic sample audio not available")
        gen = rng.choice(sorted({(r["generator"], r["speaker"]) for r in spoof}))
        pick = [config.ROOT / r["path"] for r in spoof if (r["generator"], r["speaker"]) == tuple(gen)]
        return _concat(list(rng.permutation(pick)), s, rng)

    if scenario == "genuine":
        return genuine_audio(target_s)
    if scenario == "synthetic":
        return synthetic_audio(target_s)
    if scenario == "mixed":
        return np.concatenate([genuine_audio(target_s * 0.45), synthetic_audio(target_s * 0.55)])
    raise ValueError("scenario must be genuine | synthetic | mixed")


def twilio_messages(audio16k: np.ndarray, token: str, call_sid: str | None = None, loss: float = 0.0,
                    seed: int = 0):
    """Yields Twilio Media Stream messages (dicts). `loss`: fraction of 20 ms frames dropped (timestamps keep going)."""
    rng = np.random.default_rng(seed)
    mu = encode(resample(audio16k, 16000, 8000))
    sid = call_sid or f"CAsim{uuid.uuid4().hex[:27]}"
    stream = f"MZsim{uuid.uuid4().hex[:27]}"
    yield {"event": "connected", "protocol": "Call", "version": "1.0.0"}
    yield {"event": "start", "sequenceNumber": "1", "streamSid": stream,
           "start": {"streamSid": stream, "callSid": sid, "tracks": ["inbound"], "customParameters": {"token": token},
                     "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}}}
    for k in range(len(mu) // FRAME):
        if loss and rng.random() < loss:
            continue
        yield {"event": "media", "sequenceNumber": str(k + 2), "streamSid": stream,
               "media": {"track": "inbound", "chunk": str(k + 1), "timestamp": str(k * 20),
                         "payload": base64.b64encode(mu[k * FRAME:(k + 1) * FRAME]).decode()}}
    yield {"event": "stop", "sequenceNumber": str(len(mu) // FRAME + 2), "streamSid": stream,
           "stop": {"callSid": sid}}


async def paced(messages, realtime: bool = True):
    """Async iterator releasing media frames at real-time pace (absolute schedule, 100 ms granularity)."""
    t0, k = time.monotonic(), 0
    for m in messages:
        if m["event"] == "media" and realtime:
            k = int(m["media"]["timestamp"]) // 20
            if k % 5 == 0:
                await asyncio.sleep(max(0.0, t0 + k * 0.02 - time.monotonic()))
        yield m


async def run_in_process(audio16k: np.ndarray, spec: dict, realtime: bool = True, loss: float = 0.0) -> str | None:
    """Simulate a call through the real TwilioStream path. Returns the session id."""
    from app.runtime import RT
    from app.telephony.media import TwilioStream

    token = uuid.uuid4().hex
    RT.pending[token] = spec
    ts = TwilioStream()
    try:
        async for m in paced(twilio_messages(audio16k, token, loss=loss), realtime):
            if not await ts.handle(m):
                break
    finally:
        await ts.close()
    return ts.session.id if ts.session else None


async def _cli(a):
    import websockets

    audio = load_audio(a.wav)[0] if a.wav else scenario_audio(a.scenario, a.seed)
    async with websockets.connect(a.url) as ws:
        async for m in paced(twilio_messages(audio, a.token, loss=a.loss), not a.fast):
            await ws.send(json.dumps(m))
    print(f"sent {len(audio) / 16000:.1f} s of audio")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--token", required=True, help="stream token from POST /guardian/simulate?transport=external")
    ap.add_argument("--scenario", default="synthetic", choices=["genuine", "synthetic", "mixed"])
    ap.add_argument("--wav", help="send this file instead of a scenario")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--loss", type=float, default=0.0)
    ap.add_argument("--fast", action="store_true", help="do not pace in real time")
    asyncio.run(_cli(ap.parse_args()))
