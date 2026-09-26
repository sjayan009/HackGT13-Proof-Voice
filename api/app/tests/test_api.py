"""Decode, normalisation, score range, model reload, temporal aggregation, pipeline routing, file upload E2E,
WebSocket streaming and Grok fallback — against the tiny test detector."""
from __future__ import annotations

import base64
import io
import json

import numpy as np
import pytest
import soundfile as sf


def wav_bytes(x, sr=16000, subtype="PCM_16", fmt="WAV"):
    b = io.BytesIO()
    sf.write(b, x, sr, subtype=subtype, format=fmt)
    return b.getvalue()


# ---------- ingestion ----------
def test_decode_and_resample_to_16k(tmp_path, speechlike):
    from app.audio.ingest import load_audio

    x48 = np.repeat(speechlike, 3)  # crude 48 kHz version
    stereo = np.stack([x48, x48], 1)
    p = tmp_path / "s.wav"
    sf.write(p, stereo, 48000)
    y, sr, dec = load_audio(p)
    assert sr == 48000 and dec == "soundfile"
    assert y.dtype == np.float32 and y.ndim == 1
    assert abs(len(y) - len(speechlike)) < 50  # 16 kHz output
    assert np.abs(y).max() <= 1.0


def test_decode_ffmpeg_fallback_mp3(tmp_path, speechlike):
    import subprocess

    from app.audio.ingest import ffmpeg_exe, load_audio

    src = tmp_path / "a.wav"
    sf.write(src, speechlike, 16000)
    mp4 = tmp_path / "a.m4a"
    subprocess.run([ffmpeg_exe(), "-v", "error", "-y", "-i", str(src), "-c:a", "aac", str(mp4)], check=True)
    y, _, dec = load_audio(mp4)
    assert dec == "ffmpeg" and abs(len(y) - len(speechlike)) < 16000 * 0.2


def test_pcm_bytes():
    from app.audio.ingest import pcm_bytes_to_float

    s16 = (np.array([0, 16384, -32768], "<i2")).tobytes()
    assert np.allclose(pcm_bytes_to_float(s16, "s16le"), [0, 0.5, -1.0])
    f = np.array([0.25, -0.5], "<f4").tobytes()
    assert np.allclose(pcm_bytes_to_float(f + b"\x00", "f32le"), [0.25, -0.5])


def test_normalize_ignores_padding():
    import torch

    from app.detectors.ssl_model import normalize

    x = torch.zeros(2, 100)
    x[0, :50] = torch.randn(50) * 3 + 2
    x[1] = torch.randn(100)
    y = normalize(x, torch.tensor([50, 100]))
    assert torch.allclose(y[0, 50:], torch.zeros(50))
    assert abs(float(y[0, :50].mean())) < 1e-4 and abs(float(y[0, :50].std(unbiased=False)) - 1) < 1e-3


# ---------- detector ----------
def test_score_range_and_reload_determinism(tiny_model_dir, detector, speechlike):
    from app.detectors.primary import PrimaryDetector

    r1 = detector.score(speechlike)
    assert 0.0 <= r1.synthetic_probability <= 1.0 and np.isfinite(r1.raw_score)
    r2 = PrimaryDetector(tiny_model_dir, "cpu").score(speechlike)
    assert r1.raw_score == pytest.approx(r2.raw_score, abs=1e-3)  # fp16 weights on disk -> tiny tolerance


def test_batch_padding_invariance(detector, speechlike):
    a = detector.logits([speechlike[:40000]])[0]
    b = detector.logits([speechlike[:40000], speechlike])[0]
    assert a == pytest.approx(b, abs=1e-3)


def test_windows_cover_clip(detector, speechlike):
    w = detector.score_windows(speechlike, 2000, 500)
    assert w[0]["start_ms"] == 0 and w[-1]["end_ms"] == int(len(speechlike) / 16)
    assert all(0 <= x["synthetic_probability"] <= 1 for x in w)
    assert [x["start_ms"] for x in w] == sorted(x["start_ms"] for x in w)


# ---------- temporal aggregation / orchestration ----------
def test_regions_and_ttc():
    from app.orchestration.pipeline import suspicious_regions, time_to_confidence

    ws = [{"start_ms": i * 500, "end_ms": i * 500 + 2000, "synthetic_probability": p}
          for i, p in enumerate([0.1, 0.2, 0.9, 0.95, 0.3, 0.8, 0.85])]
    regs = suspicious_regions(ws, 0.5)
    assert [(r["start_ms"], r["end_ms"]) for r in regs] == [(1000, 3500), (2500, 5000)]
    assert regs[0]["peak_probability"] == 0.95
    assert time_to_confidence(ws, 0.5, 0.9) == 4500  # decisions agree with final from window 5 onward
    assert time_to_confidence(ws, 0.5, 0.1) is None


def test_pipeline_report_schema(detector, speechlike, tmp_path):
    from app.orchestration.pipeline import analyze

    p = tmp_path / "x.wav"
    sf.write(p, speechlike, 16000)
    r = analyze(speechlike, detector, path=str(p), file_name="x.wav", orig_sr=16000)
    for k in ["synthetic_probability", "cm_score", "analysis_confidence", "status", "timeline", "suspicious_regions",
              "techniques_run", "why_run", "evidence", "branch_log", "file", "disclaimer"]:
        assert k in r
    assert 0 <= r["cm_score"] <= 1 and r["status"] in {"insufficient_evidence", "likely_human", "inconclusive",
                                                         "likely_synthetic"}
    assert {"metadata", "quality", "primary_detector"} <= set(r["techniques_run"])
    assert set(r["why_run"]) == set(r["techniques_run"])
    logged = {b["technique"] for b in r["branch_log"]}
    assert {"spectral", "prosody", "compression", "splice"} <= logged  # every branch is either run or skipped w/ reason
    assert all(b["reason"] for b in r["branch_log"])
    assert r["evidence"]["primary_detector"]["used_in_score"] is True
    json.dumps(r)  # serialisable


# ---------- HTTP / WebSocket ----------
@pytest.fixture
def client(tiny_model_dir, monkeypatch):
    from fastapi.testclient import TestClient

    from app import config

    monkeypatch.setattr(config, "MODEL_DIR", tiny_model_dir)
    monkeypatch.setattr(config, "DEVICE", "cpu")
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["detector"]["name"] == "tiny-test"


def test_file_upload_e2e(client, speechlike):
    r = client.post("/analyze/file", files={"file": ("clip.wav", wav_bytes(speechlike), "audio/wav")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["file"]["name"] == "clip.wav" and j["file"]["sample_rate"] == 16000 and 0 <= j["cm_score"] <= 1


def test_file_upload_rejects_garbage(client):
    r = client.post("/analyze/file", files={"file": ("x.wav", b"not audio at all", "audio/wav")})
    assert r.status_code == 400


def test_websocket_stream(client, speechlike):
    x48 = np.interp(np.arange(len(speechlike) * 3) / 3, np.arange(len(speechlike)), speechlike).astype("<f4")
    with client.websocket_connect("/analyze/stream") as ws:
        ws.send_text(json.dumps({"type": "start", "sample_rate": 48000, "encoding": "f32le", "source": "mic"}))
        assert ws.receive_json()["type"] == "ready"
        for i in range(0, len(x48), 4800):
            ws.send_bytes(x48[i:i + 4800].tobytes())
        ws.send_text(json.dumps({"type": "stop"}))
        events = []
        while True:
            e = ws.receive_json()
            if e["type"] != "analysis.window":
                final = e
                break
            events.append(e)
    assert len(events) >= 3  # 3.5 s of audio -> windows every 0.5 s from 1.5 s
    assert all(e["type"] == "analysis.window" for e in events)
    assert [e["t_ms"] for e in events] == sorted(e["t_ms"] for e in events)
    assert all(0 <= e["rolling_probability"] <= 1 for e in events)
    assert final["type"] == "analysis.final" and "timeline" in final["report"]


def test_redteam_falls_back_to_cached_fixture(client, monkeypatch):
    from app.services import grok

    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(grok, "_xai_api_key", lambda: None)  # no network in tests
    r = client.post("/redteam/grok", json={"text": None})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["source"] == "cached_fixture" and j["report"]["cm_score"] is not None
    assert base64.b64decode(j["audio_wav_b64"])[:4] == b"RIFF"


def test_eval_summary_404_or_json(client):
    r = client.get("/eval/summary")
    assert r.status_code in (200, 404)
