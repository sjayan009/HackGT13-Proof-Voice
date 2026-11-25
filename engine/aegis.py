"""ProofVoice Aegis — uncertainty-aware, deadline-constrained GPU scheduling for streaming deepfake detection.

Two independent decisions, so each contribution can be ablated:

1. Gating (per call, per 0.5 s hop) — *whether* a window is worth a model forward:
     fixed-rate policies infer every hop; `adaptive` skips silent hops, decimates hops on calls whose recent
     evidence is clearly below threshold, and escalates (tighter deadline) hops whose evidence is ambiguous.
2. Dispatch (one GPU worker thread) — *how* queued windows run:
     fifo      serial, arrival order, batch 1                  (the original ProofVoice behaviour)
     dynamic   microbatching up to max_batch / max_wait_ms     (ordinary dynamic batching)
     edf       microbatching + earliest-deadline-first + per-call coalescing (a newer hop supersedes a queued
               older one) + shedding of windows already past their deadline
     adaptive  edf dispatch + adaptive gating

Optional two-stage screening (`adaptive` only, when a screening backend is configured): a distilled 4-layer
student scores the window first; if it is confidently below threshold (screen llr < SCREEN_CLEAR_LLR) its
evidence is used and the 12-layer model is skipped, otherwise the window escalates to the main model.

Objective (PRD §7): minimise E[C_gpu + λ·D] subject to limits on missed detections, false alerts and abstention.
The gate thresholds below are deterministic and documented; benchmarks/policy_replay.py measures the trade-off.
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field

import numpy as np

POLICIES = ("fifo", "dynamic", "edf", "adaptive")

# adaptive gate parameters (log-odds units relative to the decision threshold)
SILENT_HOP_SPEECH_S = 0.1   # newest hop has < 100 ms of speech -> nothing new to score
CLEAR_LLR = -2.0            # last 3 inferred windows all below this -> decimate
CLEAR_STRIDE = 3            # ...to one inference every 3 hops (1.5 s)
AMBIGUOUS_LLR = 1.0         # |llr| below this, or sign flips -> escalate
SCREEN_CLEAR_LLR = -2.0     # two-stage: student llr below this -> accept the student, skip the main model


def decide(policy: str, hop_speech_s: float, state: str, recent_llr: list[float], hops_since_infer: int) -> str:
    """Gate for one hop: 'infer' | 'escalate' | 'skip_silence' | 'skip_clear'."""
    if policy != "adaptive":
        return "infer"
    if hop_speech_s < SILENT_HOP_SPEECH_S:
        return "skip_silence"
    last = recent_llr[-4:]
    if state == "no_suspicious_evidence" and len(last) >= 3 and max(last[-3:]) < CLEAR_LLR \
            and hops_since_infer < CLEAR_STRIDE:
        return "skip_clear"
    if state != "suspicious" and last and (abs(last[-1]) < AMBIGUOUS_LLR or (min(last) < 0 < max(last))):
        return "escalate"
    return "infer"


@dataclass(order=True)
class _Req:
    deadline: float
    submitted: float = field(compare=False)
    call_id: str = field(compare=False)
    clip: np.ndarray = field(compare=False, repr=False)
    fut: asyncio.Future = field(compare=False, repr=False)
    model: str = field(default="main", compare=False)


class Aegis:
    def __init__(self, backend, telemetry, policy: str = "adaptive", max_batch: int = 16, max_wait_ms: float = 5.0,
                 deadline_ms: float = 1000.0, screen=None):
        self.backend, self.tm = backend, telemetry
        self.backends = {"main": backend, "screen": screen}
        self.max_batch, self.max_wait, self.deadline = max_batch, max_wait_ms / 1000, deadline_ms / 1000
        self.set_policy(policy)
        self._q: list[_Req] = []
        self._cv = threading.Condition()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True, name="aegis-gpu")
        self._t.start()

    def set_policy(self, policy: str) -> None:
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self.policy = policy

    @property
    def queue_depth(self) -> int:
        return len(self._q)

    def decide(self, **kw) -> str:
        return decide(self.policy, **kw)

    @property
    def screening(self) -> bool:
        return self.policy == "adaptive" and self.backends["screen"] is not None

    async def infer(self, call_id: str, clip: np.ndarray, escalate: bool = False, model: str = "main") -> float | None:
        """Logit for one window, or None if the request was shed/superseded. Raises if the backend fails."""
        self._loop = self._loop or asyncio.get_running_loop()
        now = time.perf_counter()
        fut = self._loop.create_future()
        req = _Req(now + self.deadline * (0.5 if escalate else 1.0), now, call_id, clip, fut, model)
        with self._cv:
            if self.policy in ("edf", "adaptive"):  # coalesce: only the newest window of a call matters
                for old in [r for r in self._q if r.call_id == call_id]:
                    self._q.remove(old)
                    self._resolve(old.fut, None)
            self._q.append(req)
            self._cv.notify()
        return await fut

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    def _resolve(self, fut, value=None, exc: BaseException | None = None):
        def _set():
            if not fut.done():
                fut.set_exception(exc) if exc else fut.set_result(value)
        self._loop.call_soon_threadsafe(_set)

    def _take(self) -> list[_Req]:
        with self._cv:
            while not self._q and not self._stop:
                self._cv.wait()
            if self._stop:
                return []
            if self.policy != "fifo":  # wait (bounded) for the batch to fill, measured from the oldest arrival
                while len(self._q) < self.max_batch and not self._stop:
                    left = self.max_wait - (time.perf_counter() - min(r.submitted for r in self._q))
                    if left <= 0:
                        break
                    self._cv.wait(left)
            now = time.perf_counter()
            if self.policy in ("edf", "adaptive"):
                for r in [r for r in self._q if r.deadline < now]:
                    self._q.remove(r)
                    self._resolve(r.fut, None)
                self._q.sort()
            else:
                self._q.sort(key=lambda r: r.submitted)
            n = 1 if self.policy == "fifo" else self.max_batch
            head = self._q[0].model if self._q else "main"  # one model per microbatch, chosen by the most urgent
            batch = [r for r in self._q if r.model == head][:n]
            self._q = [r for r in self._q if r not in batch]
            return batch

    def _run(self):
        while not self._stop:
            batch = self._take()
            if not batch:
                continue
            t0 = time.perf_counter()
            try:
                logits = self.backends[batch[0].model].logits([r.clip for r in batch])
            except Exception as e:  # noqa: BLE001 - surfaced to the call as analysis_unavailable
                for r in batch:
                    self._resolve(r.fut, exc=e)
                continue
            t1 = time.perf_counter()
            self.tm.batch_done(len(batch), (t1 - t0) * 1000)
            for r, lg in zip(batch, logits):
                self.tm.request_done((t1 - r.submitted) * 1000, (t0 - r.submitted) * 1000)
                self._resolve(r.fut, float(lg))


if __name__ == "__main__":
    assert decide("fifo", 0.0, "inconclusive", [], 0) == "infer"
    assert decide("adaptive", 0.0, "inconclusive", [], 0) == "skip_silence"
    assert decide("adaptive", 0.5, "no_suspicious_evidence", [-3, -3, -3], 1) == "skip_clear"
    assert decide("adaptive", 0.5, "no_suspicious_evidence", [-3, -3, -3], 3) == "infer"
    assert decide("adaptive", 0.5, "inconclusive", [2.0, -1.5], 0) == "escalate"
    assert decide("adaptive", 0.5, "inconclusive", [3.0, 3.5], 0) == "infer"

    class _Echo:  # fake backend: logit = mean of clip; counts batches
        calls = []

        def logits(self, clips):
            self.calls.append(len(clips))
            time.sleep(0.01)
            return [float(c.mean()) for c in clips]

    from engine.telemetry import Telemetry

    async def main():
        for pol in POLICIES:
            be, tm = _Echo(), Telemetry()
            be.calls.clear()
            ag = Aegis(be, tm, pol, max_batch=8, max_wait_ms=20)
            out = await asyncio.gather(*[ag.infer(f"c{i}", np.full(10, float(i))) for i in range(8)])
            assert out == [float(i) for i in range(8)], (pol, out)
            assert (max(be.calls) == 1) == (pol == "fifo"), (pol, be.calls)
            # coalescing: two windows of one call queued together -> older superseded under edf/adaptive
            r = await asyncio.gather(ag.infer("x", np.ones(4)), ag.infer("x", np.zeros(4)))
            assert (r[0] is None) == (pol in ("edf", "adaptive")), (pol, r)
            ag.stop()
        be, sc = _Echo(), _Echo()
        ag = Aegis(be, Telemetry(), "adaptive", max_batch=8, max_wait_ms=20, screen=sc)
        assert ag.screening
        r = await asyncio.gather(ag.infer("a", np.ones(4), model="screen"), ag.infer("b", np.zeros(4)))
        assert r == [1.0, 0.0], r
        ag.stop()
        print("aegis ok")

    asyncio.run(main())
