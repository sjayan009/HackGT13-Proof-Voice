"""Tests for api.app.services.grok — no network access.

Covers:
- fallback to the cached fixture when the API key is missing
- fallback to the cached fixture when the realtime/TTS connections fail
- chunk ordering / seq correctness
- the API key never appears in exception text or object reprs
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.app.services import grok as g  # noqa: E402


FAKE_KEY = "xai-THIS-IS-A-FAKE-TEST-KEY-0123456789"


def run(coro):
    return asyncio.run(coro)


def _collect(agen):
    async def _inner():
        return [chunk async for chunk in agen]

    return run(_inner())


@pytest.fixture(autouse=True)
def _fixture_wav_exists():
    assert g._FIXTURE_WAV.exists(), "cached fixture wav must exist for these tests"


def test_no_key_falls_back_to_cached_fixture(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    chunks = _collect(g.stream_grok_speech("hello world"))

    assert len(chunks) > 0
    assert all(c.source == "cached_fixture" for c in chunks)


def test_connection_failure_falls_back_to_cached_fixture(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", FAKE_KEY)

    async def _boom_voice(text, voice):
        raise g.GrokUnavailable("simulated realtime connection failure")
        yield  # pragma: no cover - make this an async generator

    async def _boom_tts(text, voice):
        raise g.GrokUnavailable("simulated tts connection failure")
        yield  # pragma: no cover

    monkeypatch.setattr(g, "_stream_grok_voice", _boom_voice)
    monkeypatch.setattr(g, "_stream_grok_tts", _boom_tts)

    chunks = _collect(g.stream_grok_speech("hello world"))

    assert len(chunks) > 0
    assert all(c.source == "cached_fixture" for c in chunks)


def test_chunk_ordering_and_seq(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    chunks = _collect(g.stream_grok_speech(None))

    assert len(chunks) > 1
    seqs = [c.seq for c in chunks]
    assert seqs == sorted(seqs)
    assert seqs == list(range(len(chunks)))


def test_generate_grok_speech_uses_cached_fixture(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    waveform, sr, source, text = run(g.generate_grok_speech("a short test line"))

    assert source == "cached_fixture"
    assert sr > 0
    assert waveform.dtype.name == "float32"
    assert waveform.size > 0
    # float32 PCM16-derived samples must be within [-1, 1]
    assert waveform.max() <= 1.0
    assert waveform.min() >= -1.0


def test_grok_unavailable_message_redacts_api_key(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", FAKE_KEY)

    class _FakeExc(Exception):
        pass

    exc = _FakeExc(f"connection refused, key was {FAKE_KEY} in the header")
    wrapped = g._safe_error("test failure", exc)

    assert FAKE_KEY not in str(wrapped)
    assert "REDACTED" in str(wrapped)


def test_grok_audio_chunk_repr_never_contains_pcm_or_key():
    chunk = g.GrokAudioChunk(
        pcm16=b"\x00\x01" * 100 + FAKE_KEY.encode(),
        sample_rate=24000,
        seq=0,
        source="cached_fixture",
    )
    text = repr(chunk)

    assert FAKE_KEY not in text
    assert "bytes" in text


def test_no_key_explain_report_raises_grok_unavailable(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    with pytest.raises(g.GrokUnavailable):
        run(g.explain_report({"synthetic_probability": 0.9}))


def test_explain_report_error_never_leaks_key(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", FAKE_KEY)

    class _FakeResponse:
        status_code = 500
        text = f"internal error, saw key {FAKE_KEY}"

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, *args, **kwargs):
            return _FakeResponse()

    monkeypatch.setattr(g.httpx, "AsyncClient", lambda *a, **kw: _FakeClient())

    with pytest.raises(g.GrokUnavailable) as excinfo:
        run(g.explain_report({"synthetic_probability": 0.9}))

    assert FAKE_KEY not in str(excinfo.value)
