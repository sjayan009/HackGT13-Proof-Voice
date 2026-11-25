"""Twilio Programmable Voice webhooks + Media Streams WebSocket (PRD §4 MVP call flow).

Call flow:  caller -> dedicated ProofVoice number -> /twilio/voice (disclosure + consent prompt)
            -> /twilio/consent: consent => <Start><Stream> forks the caller's audio to /twilio/media, then <Dial>s the
               protected recipient; no consent => the call is connected without analysis.
Inference never sits in the conversation's audio path: if ProofVoice fails, the call carries on.

Security: webhooks are verified with X-Twilio-Signature when TWILIO_AUTH_TOKEN is set. The media WebSocket accepts
only streams that present a single-use token minted by a verified /twilio/consent request.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import logging
import secrets
from xml.sax.saxutils import escape, quoteattr

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from app import config
from app.runtime import RT
from app.telephony.media import TwilioStream
from app.verification.alerts import DEV_USER, dev_mode

log = logging.getLogger("proofvoice.twilio")
router = APIRouter(prefix="/twilio", tags=["telephony"])

DISCLOSURE = ("This call is protected by ProofVoice Guardian. With your permission, the caller's voice will be "
              "checked in real time for signs of A I generated speech. Audio is not recorded or stored. "
              "Press 1 to allow this and connect. Press 2 to connect without the check.")


def signature(url: str, params: dict, token: str) -> str:
    """Twilio request signature: base64(HMAC-SHA1(token, url + concat(sorted key+value)))."""
    data = url + "".join(k + params[k] for k in sorted(params))
    return base64.b64encode(hmac.new(token.encode(), data.encode(), hashlib.sha1).digest()).decode()


def public_url(request: Request) -> str:
    base = config.PUBLIC_BASE_URL or str(request.base_url).rstrip("/")
    q = request.url.query
    return f"{base}{request.url.path}" + (f"?{q}" if q else "")


async def verified_form(request: Request) -> dict:
    form = {k: str(v) for k, v in (await request.form()).items()}
    if config.TWILIO_AUTH_TOKEN:
        expected = signature(public_url(request), form, config.TWILIO_AUTH_TOKEN)
        if not hmac.compare_digest(expected, request.headers.get("X-Twilio-Signature", "")):
            log.warning("rejected webhook with bad Twilio signature (check PUBLIC_BASE_URL matches the ngrok URL)")
            raise HTTPException(403, "invalid Twilio signature")
    return form


def twiml(body: str) -> Response:
    return Response(f'<?xml version="1.0" encoding="UTF-8"?><Response>{body}</Response>', media_type="text/xml")


def same_phone(a: str, b: str) -> bool:
    """Same phone number, ignoring formatting (+1 (404) 555-0123 == 4045550123)."""
    da, db = (re.sub(r"\D", "", x)[-10:] for x in (a, b))
    return len(da) == 10 and da == db


async def line_for(number: str) -> dict | None:
    line = await RT.store.line_by_number(number) if number else None
    if line is None and dev_mode() and config.GUARDIAN_RECIPIENT_NUMBER:
        line = {"id": None, "user_id": DEV_USER, "twilio_number": number,
                "recipient_phone": config.GUARDIAN_RECIPIENT_NUMBER}
    return line


@router.post("/voice")
async def voice(request: Request):
    f = await verified_form(request)
    line = await line_for(f.get("To", ""))
    if not line:
        return twiml("<Say>This ProofVoice number is not set up yet. Goodbye.</Say><Hangup/>")
    return twiml(f'<Gather numDigits="1" timeout="8" action="/twilio/consent" method="POST">'
                 f"<Say>{escape(DISCLOSURE)}</Say></Gather><Redirect method=\"POST\">/twilio/consent</Redirect>")


@router.post("/consent")
async def consent(request: Request):
    f = await verified_form(request)
    line = await line_for(f.get("To", ""))
    if not line:
        return twiml("<Hangup/>")
    granted = f.get("Digits") == "1"
    spec = {"user_id": line["user_id"], "caller": f.get("From"), "line": line,
            "consent": "granted" if granted else "declined"}
    # no callerId: Twilio passes the original caller's number through, so the recipient sees who is calling
    dial = f'<Dial action="/twilio/status" method="POST"><Number>{escape(line["recipient_phone"])}</Number></Dial>'
    if same_phone(f.get("From", ""), line["recipient_phone"]):
        # self-test: the protected person is calling their own line, so there is nobody to connect to. Keep the
        # line open and analyse the caller; the stream's stop event ends the session when they hang up.
        dial = ("<Say>Self test mode. Talk, or play a recording into the phone. Hang up when you are done.</Say>"
                '<Pause length="600"/><Hangup/>')
    if not granted:
        from app.telephony.media import CallSession

        s = CallSession(call_sid=f.get("CallSid", ""), **spec)
        await s.end()
        return twiml(f"<Say>Connecting without the check.</Say>{dial}")
    token = secrets.token_urlsafe(24)
    RT.pending[token] = spec
    ws = (config.PUBLIC_BASE_URL or str(request.base_url).rstrip("/")).replace("https://", "wss://", 1) \
        .replace("http://", "ws://", 1)
    return twiml(f'<Say>Thank you. Connecting.</Say><Start><Stream url={quoteattr(ws + "/twilio/media")} '
                 f'track="inbound_track"><Parameter name="token" value={quoteattr(token)}/></Stream></Start>{dial}')


@router.post("/status")
async def status(request: Request):
    f = await verified_form(request)
    sid = f.get("CallSid")
    for s in list(RT.calls.values()):
        if s.call_sid == sid:
            await s.end()
    return twiml("<Hangup/>")


@router.websocket("/media")
async def media(ws: WebSocket):
    await ws.accept()
    ts = TwilioStream()
    try:
        while await ts.handle(json.loads(await ws.receive_text())):
            pass
    except WebSocketDisconnect:
        pass
    except (PermissionError, ValueError) as e:
        log.warning("media stream rejected: %s", e)
        await ws.close(code=1008)
    finally:
        await ts.close()
