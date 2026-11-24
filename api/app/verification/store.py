"""Minimal persistence for Guardian: lines, trusted contacts, call sessions, risk events. Never audio.

Supabase PostgREST with the project's secret (service-role) key when configured; otherwise an in-memory dev store.
Writes are fire-and-forget so a slow database can never stall the audio path; failures are logged, not raised.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone

import httpx

log = logging.getLogger("proofvoice.store")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, url: str, secret_key: str):
        self.enabled = bool(url and secret_key)
        self.url = f"{url}/rest/v1"
        h = {"apikey": secret_key, "Content-Type": "application/json", "Prefer": "return=minimal"}
        if secret_key and not secret_key.startswith("sb_"):  # legacy service_role JWT
            h["Authorization"] = f"Bearer {secret_key}"
        self.client = httpx.AsyncClient(headers=h, timeout=10) if self.enabled else None
        self.mem: dict[str, list[dict]] = {"guardian_lines": [], "trusted_contacts": [], "call_sessions": [],
                                           "risk_events": []}
        self._tasks: set[asyncio.Task] = set()

    async def _select(self, table: str, **eq) -> list[dict]:
        if not self.enabled:
            return [r for r in self.mem[table] if all(r.get(k) == v for k, v in eq.items())]
        r = await self.client.get(f"{self.url}/{table}", params={k: f"eq.{v}" for k, v in eq.items()} | {"select": "*"})
        r.raise_for_status()
        return r.json()

    async def line_by_number(self, number: str) -> dict | None:
        rows = await self._select("guardian_lines", twilio_number=number)
        return rows[0] if rows else None

    async def contacts(self, user_id: str) -> list[dict]:
        return await self._select("trusted_contacts", user_id=user_id)

    async def session(self, session_id: str) -> dict | None:
        rows = await self._select("call_sessions", id=session_id)
        return rows[0] if rows else None

    # --- fire-and-forget writes -------------------------------------------------------------------------------
    def _bg(self, coro) -> None:
        t = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._done)

    def _done(self, t: asyncio.Task) -> None:
        self._tasks.discard(t)
        if not t.cancelled() and t.exception():
            log.warning("store write failed: %s", t.exception())

    async def _write(self, method: str, table: str, row: dict, **eq) -> None:
        if not self.enabled:
            if method == "post":
                self.mem[table].append(dict(row))
            else:
                for r in self.mem[table]:
                    if all(r.get(k) == v for k, v in eq.items()):
                        r.update(row)
            return
        params = {k: f"eq.{v}" for k, v in eq.items()}
        r = await self.client.request(method.upper(), f"{self.url}/{table}", json=row, params=params)
        r.raise_for_status()

    def create_session(self, row: dict) -> str:
        row = {"started_at": now_iso()} | row | {"id": row.get("id") or str(uuid.uuid4())}
        if row.get("user_id"):
            self._bg(self._write("post", "call_sessions", row))
        return row["id"]

    def end_session(self, session_id: str, user_id: str | None, patch: dict) -> None:
        if user_id:
            self._bg(self._write("patch", "call_sessions", {"ended_at": now_iso()} | patch, id=session_id))

    def add_event(self, session_id: str, user_id: str | None, kind: str, **fields) -> None:
        if user_id:
            self._bg(self._write("post", "risk_events", {"session_id": session_id, "user_id": user_id, "kind": kind}
                                 | fields))

    async def drain(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
