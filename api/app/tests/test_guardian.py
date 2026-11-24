"""v2 Guardian: Twilio webhooks/signatures, Media Streams -> live dashboard events, consent, packet loss, engine API.
Runs against the tiny random detector in dev mode (no Supabase, no Twilio, no network)."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def gclient(tiny_model_dir, monkeypatch):
    from fastapi.testclient import TestClient

    from app import config
    from app.main import app

    monkeypatch.setattr(config, "MODEL_DIR", tiny_model_dir)
    monkeypatch.setattr(config, "DEVICE", "cpu")
    for k in ("SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY", "SUPABASE_SECRET_KEY", "TWILIO_AUTH_TOKEN",
              "TWILIO_ACCOUNT_SID", "PUBLIC_BASE_URL"):
        monkeypatch.setattr(config, k, "")
    monkeypatch.setattr(config, "GUARDIAN_RECIPIENT_NUMBER", "+15550000002")
    with TestClient(app) as c:
        yield c


def call_audio(speechlike, seconds=6.0):
    """Tiled harmonic signal gated into 'syllables' (0.3 s on / 0.15 s off) so the energy VAD sees speech."""
    x = np.tile(speechlike, int(np.ceil(seconds / 3.5)))[: int(seconds * 16000)]
    on = (np.arange(len(x)) % 7200) < 4800
    return (x * np.where(on, 1.0, 0.01)).astype(np.float32)


def test_twilio_signature_matches_twilio_docs_example():
    from app.telephony.twilio import signature

    params = {"CallSid": "CA1234567890ABCDE", "Caller": "+12349013030", "Digits": "1234",
              "From": "+12349013030", "To": "+18005551212"}
    assert signature("https://mycompany.com/myapp.php?foo=1&bar=2", params, "12345") == "0/KCTR6DLpKmkAf8muzZqo1nDgQ="


def test_webhook_rejects_bad_signature(gclient, monkeypatch):
    from app import config
    from app.telephony.twilio import signature

    monkeypatch.setattr(config, "TWILIO_AUTH_TOKEN", "secret")
    form = {"To": "+15550000001", "From": "+15550000003", "CallSid": "CA1"}
    assert gclient.post("/twilio/voice", data=form, headers={"X-Twilio-Signature": "nope"}).status_code == 403
    good = signature("http://testserver/twilio/voice", form, "secret")
    r = gclient.post("/twilio/voice", data=form, headers={"X-Twilio-Signature": good})
    assert r.status_code == 200 and "<Gather" in r.text and "not recorded" in r.text


def test_consent_forks_stream_and_dials_recipient(gclient):
    from app.runtime import RT

    r = gclient.post("/twilio/consent", data={"To": "+15550000001", "From": "+15550000003", "CallSid": "CA2",
                                              "Digits": "1"})
    assert "<Stream" in r.text and 'track="inbound_track"' in r.text and "+15550000002</Number>" in r.text
    assert len(RT.pending) == 1 and next(iter(RT.pending.values()))["consent"] == "granted"
    RT.pending.clear()


def test_declined_consent_connects_without_analysis(gclient):
    with gclient.websocket_connect("/guardian/live") as live:
        assert live.receive_json()["type"] == "hello"
        r = gclient.post("/twilio/consent", data={"To": "+1555", "From": "+1556", "CallSid": "CA3", "Digits": "2"})
        assert "<Stream" not in r.text and "<Dial" in r.text
        evs = [live.receive_json() for _ in range(3)]
    assert evs[0]["type"] == "call.started" and evs[0]["consent"] == "declined"
    assert evs[1]["state"] == "analysis_unavailable" and evs[2]["type"] == "call.ended"


def _run_media_call(gclient, audio, loss=0.0):
    from app.runtime import RT
    from app.telephony.sim import twilio_messages

    RT.pending["tok"] = {"user_id": "dev-user", "caller": "+15550000003", "consent": "granted", "line": None}
    with gclient.websocket_connect("/guardian/live") as live:
        live.receive_json()
        with gclient.websocket_connect("/twilio/media") as ws:
            for m in twilio_messages(audio, "tok", loss=loss, seed=1):
                ws.send_json(m)
        evs = []
        while not evs or evs[-1]["type"] != "call.ended":
            evs.append(live.receive_json())
    return evs


def test_media_stream_end_to_end(gclient, speechlike):
    evs = _run_media_call(gclient, call_audio(speechlike, 6.0))
    assert evs[0]["type"] == "call.started" and evs[0]["consent"] == "granted"
    ups = [e for e in evs if e["type"] == "call.update"]
    assert len(ups) >= 5  # first window at 3 s, then one per 0.5 s hop
    inferred = [u for u in ups if u["window"]]
    assert inferred, ups
    w = inferred[0]["window"]
    assert 0 <= w["synthetic_probability"] <= 1 and w["end_ms"] - w["start_ms"] == 3000
    assert all(u["state"] in ("no_suspicious_evidence", "inconclusive", "suspicious", "analysis_unavailable")
               for u in ups)
    end = evs[-1]
    assert end["windows_inferred"] + end["windows_skipped"] == len(ups)


def test_packet_loss_is_measured(gclient, speechlike):
    evs = _run_media_call(gclient, call_audio(speechlike, 6.0), loss=0.2)
    losses = [e["window"]["packet_loss"] for e in evs if e["type"] == "call.update" and e["window"]]
    assert losses and max(losses) > 0.05


def test_media_stream_rejects_unknown_token(gclient, speechlike):
    from starlette.websockets import WebSocketDisconnect

    from app.telephony.sim import twilio_messages

    with pytest.raises(WebSocketDisconnect) as e:
        with gclient.websocket_connect("/twilio/media") as ws:
            for m in twilio_messages(call_audio(speechlike, 1.0), "forged"):
                ws.send_json(m)
            ws.receive_json()
    assert e.value.code == 1008


def test_engine_metrics_policy_and_prometheus(gclient):
    m = gclient.get("/engine/metrics").json()
    assert m["policy"] == "adaptive" and m["backend"] == "pytorch-fp32" and m["gpu"] is None
    assert set(m["actions"]) == {"infer", "escalate", "skip_silence", "skip_clear", "shed"}
    assert gclient.post("/engine/policy", json={"policy": "fifo"}).json()["policy"] == "fifo"
    assert gclient.post("/engine/policy", json={"policy": "bogus"}).status_code == 422
    assert "proofvoice_hops_total" in gclient.get("/metrics").text
    assert gclient.get("/health").json()["guardian"]["dev_mode"] is True


def test_verification_requires_owned_session(gclient):
    r = gclient.post("/guardian/sessions/nope/verification", json={"outcome": "verified_genuine"})
    assert r.status_code == 404
