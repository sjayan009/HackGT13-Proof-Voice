"""xAI / Grok Voice integration for ProofVoice's red-team workstream.

Grok is used ONLY to generate synthetic speech samples that the ProofVoice forensic
pipeline then analyzes. Grok never sets a detection score.

Path priority for `stream_grok_speech`:
    1. Grok Voice realtime (wss://api.x.ai/v1/realtime) -- source="grok_voice"
    2. xAI TTS REST (https://api.x.ai/v1/tts)            -- source="grok_tts"
    3. Cached local fixture WAV                          -- source="cached_fixture"

See XAI_NOTES.md in this directory for the exact verified endpoints/events.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv should always be installed
    load_dotenv = None

try:
    import websockets
except ImportError:  # pragma: no cover
    websockets = None

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None


# ---------------------------------------------------------------------------
# Config / env loading
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_FIXTURES_DIR = _THIS_DIR.parent / "fixtures"
_FIXTURE_WAV = _FIXTURES_DIR / "grok_fixture.wav"
_FIXTURE_JSON = _FIXTURES_DIR / "grok_fixture.json"

_ENV_CANDIDATES = [
    _THIS_DIR.parent.parent / ".env",  # api/.env
    Path.cwd() / "api" / ".env",
]

if load_dotenv is not None:
    for _env_path in _ENV_CANDIDATES:
        if _env_path.exists():
            load_dotenv(_env_path, override=False)
            break

REALTIME_WS_URL = "wss://api.x.ai/v1/realtime"
TTS_URL = "https://api.x.ai/v1/tts"
CHAT_COMPLETIONS_URL = "https://api.x.ai/v1/chat/completions"

DEFAULT_SAMPLE_RATE = 24000
CONNECT_TIMEOUT_S = 10.0
TOTAL_TIMEOUT_S = 30.0

DEFAULT_DEMO_LINE = (
    "Hi, it's me, I need you to wire money right away, it's an emergency and I "
    "can't talk long."
)


def _xai_api_key() -> Optional[str]:
    return os.environ.get("XAI_API_KEY") or None


def _realtime_model() -> str:
    return os.environ.get("XAI_REALTIME_MODEL") or "grok-voice-latest"


def _explain_model() -> str:
    return os.environ.get("XAI_EXPLAIN_MODEL") or "grok-4"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class GrokUnavailable(Exception):
    """Raised when no Grok path (voice, TTS) can be reached.

    The message is guaranteed to never contain the API key.
    """


def _redact(text: str) -> str:
    key = _xai_api_key()
    if key and key in text:
        text = text.replace(key, "***REDACTED***")
    return text


def _safe_error(prefix: str, exc: Exception) -> "GrokUnavailable":
    """Build a GrokUnavailable whose text cannot contain the API key."""
    return GrokUnavailable(f"{prefix}: {_redact(str(exc))}")


def _safe_unavailable(message: str) -> "GrokUnavailable":
    """Build a GrokUnavailable from a plain message, redacting the key just in case
    it was interpolated in (e.g. from a raw server response body)."""
    return GrokUnavailable(_redact(message))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class GrokAudioChunk:
    pcm16: bytes
    sample_rate: int
    seq: int
    source: str  # "grok_voice" | "grok_tts" | "cached_fixture"

    def __repr__(self) -> str:  # never leak key material even via repr of a container
        return (
            f"GrokAudioChunk(seq={self.seq}, sample_rate={self.sample_rate}, "
            f"source={self.source!r}, pcm16=<{len(self.pcm16)} bytes>)"
        )


# ---------------------------------------------------------------------------
# Path 1: Grok Voice realtime
# ---------------------------------------------------------------------------


async def _stream_grok_voice(text: str, voice: Optional[str]) -> AsyncIterator[GrokAudioChunk]:
    if websockets is None:
        raise GrokUnavailable("websockets package not installed")
    api_key = _xai_api_key()
    if not api_key:
        raise GrokUnavailable("XAI_API_KEY not set")

    url = f"{REALTIME_WS_URL}?model={_realtime_model()}"
    headers = {"Authorization": f"Bearer {api_key}"}

    seq = 0
    try:
        connect_coro = websockets.connect(
            url,
            additional_headers=headers,
            open_timeout=CONNECT_TIMEOUT_S,
        )
        async with asyncio.timeout(TOTAL_TIMEOUT_S):
            async with await asyncio.wait_for(connect_coro, timeout=CONNECT_TIMEOUT_S) as ws:
                session_update = {
                    "type": "session.update",
                    "session": {
                        "voice": voice or "eve",
                        "instructions": (
                            "You are generating a short demo audio sample for a synthetic "
                            "speech detector. Speak the user's next message verbatim, "
                            "naturally, and then stop."
                        ),
                        "turn_detection": None,
                        "audio": {
                            "output": {
                                "format": {"type": "audio/pcm", "rate": DEFAULT_SAMPLE_RATE}
                            }
                        },
                    },
                }
                await ws.send(json.dumps(session_update))

                item_create = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    },
                }
                await ws.send(json.dumps(item_create))
                await ws.send(json.dumps({"type": "response.create"}))

                async for raw in ws:
                    if isinstance(raw, (bytes, bytearray)):
                        # Some deployments send raw binary audio frames directly.
                        if raw:
                            yield GrokAudioChunk(
                                pcm16=bytes(raw),
                                sample_rate=DEFAULT_SAMPLE_RATE,
                                seq=seq,
                                source="grok_voice",
                            )
                            seq += 1
                        continue

                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type", "")

                    if etype == "error":
                        detail = event.get("error", event)
                        raise _safe_unavailable(f"realtime error event: {detail}")

                    if etype in ("response.output_audio.delta", "response.audio.delta"):
                        b64 = event.get("delta") or event.get("audio")
                        if not b64:
                            continue
                        pcm = base64.b64decode(b64)
                        if pcm:
                            yield GrokAudioChunk(
                                pcm16=pcm,
                                sample_rate=event.get("sample_rate", DEFAULT_SAMPLE_RATE),
                                seq=seq,
                                source="grok_voice",
                            )
                            seq += 1

                    elif etype == "response.done":
                        return
    except GrokUnavailable:
        raise
    except (asyncio.TimeoutError, TimeoutError) as exc:
        raise _safe_error("Grok Voice realtime timed out", exc) from exc
    except Exception as exc:  # noqa: BLE001 - convert everything to GrokUnavailable
        raise _safe_error("Grok Voice realtime failed", exc) from exc

    if seq == 0:
        raise GrokUnavailable("Grok Voice realtime returned no audio")


# ---------------------------------------------------------------------------
# Path 2: xAI TTS REST fallback
# ---------------------------------------------------------------------------


async def _stream_grok_tts(text: str, voice: Optional[str]) -> AsyncIterator[GrokAudioChunk]:
    if httpx is None:
        raise GrokUnavailable("httpx package not installed")
    api_key = _xai_api_key()
    if not api_key:
        raise GrokUnavailable("XAI_API_KEY not set")

    payload = {
        "text": text,
        "language": "en",
        "voice_id": voice or "eve",
        "output_format": {"codec": "pcm", "sample_rate": DEFAULT_SAMPLE_RATE},
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        timeout = httpx.Timeout(TOTAL_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(TTS_URL, json=payload, headers=headers)
            if resp.status_code >= 400:
                raise _safe_unavailable(
                    f"xAI TTS HTTP {resp.status_code}: {resp.text[:200]}"
                )
            pcm = resp.content
    except GrokUnavailable:
        raise
    except (httpx.TimeoutException,) as exc:
        raise _safe_error("xAI TTS timed out", exc) from exc
    except Exception as exc:  # noqa: BLE001
        raise _safe_error("xAI TTS failed", exc) from exc

    if not pcm:
        raise GrokUnavailable("xAI TTS returned empty audio")

    yield GrokAudioChunk(pcm16=pcm, sample_rate=DEFAULT_SAMPLE_RATE, seq=0, source="grok_tts")


# ---------------------------------------------------------------------------
# Path 3: cached fixture fallback
# ---------------------------------------------------------------------------


def _read_wav_pcm16(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    return pcm, sample_rate


async def _stream_cached_fixture() -> AsyncIterator[GrokAudioChunk]:
    if not _FIXTURE_WAV.exists():
        raise GrokUnavailable(
            f"no Grok path available and cached fixture missing at {_FIXTURE_WAV.name}"
        )
    pcm, sample_rate = _read_wav_pcm16(_FIXTURE_WAV)
    # Chunk it up to look like a real stream (~4096-sample frames).
    chunk_samples = 4096
    chunk_bytes = chunk_samples * 2  # int16
    seq = 0
    for offset in range(0, len(pcm), chunk_bytes):
        yield GrokAudioChunk(
            pcm16=pcm[offset : offset + chunk_bytes],
            sample_rate=sample_rate,
            seq=seq,
            source="cached_fixture",
        )
        seq += 1
        await asyncio.sleep(0)  # cooperative yield


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def stream_grok_speech(
    text: Optional[str], voice: Optional[str] = None
) -> AsyncIterator[GrokAudioChunk]:
    """Yield GrokAudioChunk objects for `text` spoken by Grok.

    Tries Grok Voice realtime, then xAI TTS, then the cached fixture. Each chunk's
    `source` field truthfully identifies which path produced it.
    """
    spoken_text = text if text else DEFAULT_DEMO_LINE

    try:
        got_any = False
        async for chunk in _stream_grok_voice(spoken_text, voice):
            got_any = True
            yield chunk
        if got_any:
            return
    except GrokUnavailable:
        pass

    try:
        got_any = False
        async for chunk in _stream_grok_tts(spoken_text, voice):
            got_any = True
            yield chunk
        if got_any:
            return
    except GrokUnavailable:
        pass

    async for chunk in _stream_cached_fixture():
        yield chunk


async def generate_grok_speech(
    text: Optional[str], voice: Optional[str] = None
) -> tuple[np.ndarray, int, str, str]:
    """Collect the full stream into (float32 mono waveform, sample_rate, source, text)."""
    spoken_text = text if text else DEFAULT_DEMO_LINE
    pcm_parts: list[bytes] = []
    sample_rate = DEFAULT_SAMPLE_RATE
    source = "cached_fixture"

    async for chunk in stream_grok_speech(text, voice):
        pcm_parts.append(chunk.pcm16)
        sample_rate = chunk.sample_rate
        source = chunk.source

    if not pcm_parts:
        raise GrokUnavailable("no audio produced by any Grok path")

    raw = b"".join(pcm_parts)
    int16 = np.frombuffer(raw, dtype="<i2")
    waveform = (int16.astype(np.float32)) / 32768.0
    return waveform, sample_rate, source, spoken_text


# ---------------------------------------------------------------------------
# Grounded explanation (POST /explain)
# ---------------------------------------------------------------------------

_EXPLAIN_SYSTEM_PROMPT = (
    "You are a forensic report summarizer. You will be given a JSON evidence object "
    "produced by an audio synthetic-speech detector. Restate ONLY facts that are "
    "explicitly present in that JSON. Do not invent numbers, do not speculate about "
    "causes not present in the evidence, do not add caveats or claims not grounded in "
    "the JSON fields. Write 2-4 plain-English sentences a non-technical analyst can "
    "read quickly. If a field is null or missing, do not mention it."
)


async def explain_report(report: dict) -> dict:
    """Ask Grok to restate `report`'s evidence in plain English. Grounded only.

    Raises GrokUnavailable if there is no key or the network/model call fails.
    """
    if httpx is None:
        raise GrokUnavailable("httpx package not installed")
    api_key = _xai_api_key()
    if not api_key:
        raise GrokUnavailable("XAI_API_KEY not set")

    model = _explain_model()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _EXPLAIN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Evidence JSON:\n"
                    + json.dumps(report, ensure_ascii=False)
                    + "\n\nSummarize only what is present above."
                ),
            },
        ],
        "stream": False,
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        timeout = httpx.Timeout(TOTAL_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(CHAT_COMPLETIONS_URL, json=payload, headers=headers)
            if resp.status_code >= 400:
                raise _safe_unavailable(
                    f"xAI chat completions HTTP {resp.status_code}: {resp.text[:200]}"
                )
            data = resp.json()
    except GrokUnavailable:
        raise
    except (httpx.TimeoutException,) as exc:
        raise _safe_error("Grok explanation timed out", exc) from exc
    except Exception as exc:  # noqa: BLE001
        raise _safe_error("Grok explanation failed", exc) from exc

    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise _safe_error("Grok explanation returned unexpected shape", exc) from exc

    return {"text": text, "model": model, "grounded": True}
