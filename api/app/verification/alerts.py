"""Live event fan-out to Guardian dashboards, SMS alerts, and Supabase access-token verification."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time

import httpx

from app import config

log = logging.getLogger("proofvoice.alerts")
DEV_USER = "dev-user"

ALERT_MESSAGE = ("Potential synthetic speech detected. Verify independently before sharing information or "
                 "sending money.")


def dev_mode() -> bool:
    return not (config.SUPABASE_URL and config.SUPABASE_PUBLISHABLE_KEY)


class Hub:
    """user_id -> subscriber queues. Events for calls with no owner (load tests) are never published."""

    def __init__(self):
        self.subs: dict[str, set[asyncio.Queue]] = {}

    def subscribe(self, user_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self.subs.setdefault(user_id, set()).add(q)
        return q

    def unsubscribe(self, user_id: str, q: asyncio.Queue) -> None:
        self.subs.get(user_id, set()).discard(q)

    def publish(self, user_id: str | None, event: dict) -> None:
        for q in self.subs.get(user_id or "", ()):
            if q.full():  # a stalled dashboard drops its oldest event instead of blocking calls
                q.get_nowait()
            q.put_nowait(event)


_token_cache: dict[str, tuple[float, str]] = {}


async def user_from_token(token: str | None) -> str | None:
    """Supabase user id for an access token (validated by Supabase Auth, cached 60 s). Dev mode -> DEV_USER."""
    if dev_mode():
        return DEV_USER
    if not token:
        return None
    k = hashlib.sha256(token.encode()).hexdigest()
    hit = _token_cache.get(k)
    if hit and hit[0] > time.monotonic():
        return hit[1]
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{config.SUPABASE_URL}/auth/v1/user",
                        headers={"apikey": config.SUPABASE_PUBLISHABLE_KEY, "Authorization": f"Bearer {token}"})
    if r.status_code != 200:
        return None
    uid = r.json().get("id")
    if len(_token_cache) > 1000:
        _token_cache.clear()
    _token_cache[k] = (time.monotonic() + 60, uid)
    return uid


def twilio_configured() -> bool:
    return bool(config.TWILIO_ACCOUNT_SID and config.TWILIO_AUTH_TOKEN and config.TWILIO_PHONE_NUMBER)


async def send_sms(to: str, body: str) -> bool:
    """SMS via Twilio REST. US long-code SMS may additionally require A2P 10DLC registration on the account."""
    if not (twilio_configured() and config.GUARDIAN_SMS_ALERTS and to):
        return False
    url = f"https://api.twilio.com/2010-04-01/Accounts/{config.TWILIO_ACCOUNT_SID}/Messages.json"
    try:
        async with httpx.AsyncClient(timeout=10, auth=(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)) as c:
            r = await c.post(url, data={"From": config.TWILIO_PHONE_NUMBER, "To": to, "Body": body})
        if r.status_code >= 300:
            log.warning("SMS alert failed: HTTP %s %s", r.status_code, r.text[:200])
        return r.status_code < 300
    except httpx.HTTPError as e:
        log.warning("SMS alert failed: %s", e)
        return False
