"""ProofVoice API — see api/CONTRACT.md.  Run:  cd api && uvicorn app.main:app --port 8000"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app import config
from app.audio.ingest import DecodeError, load_audio, pcm_bytes_to_float
from app.orchestration.pipeline import analyze
from app.services.stream import StreamAnalyzer

log = logging.getLogger("proofvoice")
STATE: dict = {}


def get_detector():
    det = STATE.get("detector")
    if det is None:
        raise HTTPException(503, "detector not loaded: " + STATE.get("detector_error", "unknown"))
    return det


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.detectors.primary import PrimaryDetector

    try:
        STATE["detector"] = PrimaryDetector(config.MODEL_DIR, config.resolve_device())
        # warm-up so the first request is not slow
        STATE["detector"].score(np.zeros(16000, np.float32))
    except Exception as e:  # noqa: BLE001
        STATE["detector_error"] = f"{type(e).__name__}: {e}"
        log.exception("detector failed to load")
    yield


app = FastAPI(title="ProofVoice", version=config.VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS + ["*"], allow_methods=["*"],
                   allow_headers=["*"])


def _grok_available() -> bool:
    import os

    return bool(os.getenv("XAI_API_KEY"))


@app.get("/health")
def health():
    det = STATE.get("detector")
    return {"status": "ok" if det else "degraded", "version": config.VERSION,
            "detector": {"name": det.name, "version": det.version, "device": str(det.device),
                         "decision_threshold": det.threshold} if det else None,
            "detector_error": STATE.get("detector_error"), "grok_available": _grok_available()}


def _analyze_array(audio: np.ndarray, **kw) -> dict:
    audio = audio[: int(config.MAX_ANALYZE_S * 16000)]
    return analyze(audio, get_detector(), win_ms=config.WINDOW_MS, hop_ms=config.HOP_MS, **kw)


@app.post("/analyze/file")
async def analyze_file(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(400, f"file larger than {config.MAX_UPLOAD_MB} MB")
    suffix = Path(file.filename or "upload").suffix[:10] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(data)
        tmp = tf.name
    try:
        try:
            audio, sr, _ = await asyncio.to_thread(load_audio, tmp)
        except DecodeError as e:
            raise HTTPException(400, str(e)) from e
        if len(audio) < 1600:
            raise HTTPException(400, "audio shorter than 0.1 s")
        return await asyncio.to_thread(_analyze_array, audio, path=tmp, file_name=file.filename,
                                       orig_sr=sr if sr > 0 else None)
    finally:
        Path(tmp).unlink(missing_ok=True)


@app.websocket("/analyze/stream")
async def analyze_stream(ws: WebSocket):
    await ws.accept()
    det = STATE.get("detector")
    if det is None:
        await ws.send_json({"type": "error", "detail": "detector not loaded"})
        await ws.close()
        return
    try:
        start = json.loads(await ws.receive_text())
        if start.get("type") != "start":
            raise ValueError("first message must be {type:'start',...}")
        enc = start.get("encoding", "f32le")
        sa = StreamAnalyzer(det, int(start.get("sample_rate", 16000)), config.WINDOW_MS, config.HOP_MS)
        await ws.send_json({"type": "ready"})
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                return
            if msg.get("bytes") is not None:
                sa.push(pcm_bytes_to_float(msg["bytes"], enc))
                while sa.ready():
                    await ws.send_json(await asyncio.to_thread(sa.step))
            elif msg.get("text"):
                if json.loads(msg["text"]).get("type") == "stop":
                    sa.flush()
                    audio = sa.audio()
                    if len(audio) >= 8000:
                        rep = await asyncio.to_thread(_analyze_array, audio, file_name=f"live-{start.get('source', 'mic')}")
                        await ws.send_json({"type": "analysis.final", "report": rep})
                    else:
                        await ws.send_json({"type": "error", "detail": "less than 0.5 s of audio received"})
                    await ws.close()
                    return
    except WebSocketDisconnect:
        return
    except Exception as e:  # noqa: BLE001
        try:
            await ws.send_json({"type": "error", "detail": f"{type(e).__name__}: {e}"})
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


class RedteamReq(BaseModel):
    text: str | None = None
    voice: str | None = None


def _wav_b64(x: np.ndarray, sr: int) -> str:
    import soundfile as sf

    b = io.BytesIO()
    sf.write(b, x, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(b.getvalue()).decode()


@app.post("/redteam/grok")
async def redteam_grok(req: RedteamReq):
    from app.audio.ingest import resample
    from app.services.grok import generate_grok_speech

    audio, sr, source, text = await generate_grok_speech(req.text, req.voice)
    a16 = resample(audio, sr)
    rep = await asyncio.to_thread(_analyze_array, a16, file_name=f"grok-{source}.wav", orig_sr=sr)
    note = {"grok_voice": "Generated live by Grok Voice (realtime API).",
            "grok_tts": "Grok Voice realtime unavailable; generated live by xAI TTS.",
            "cached_fixture": "xAI unreachable; replaying a previously generated Grok Voice clip."}.get(source, "")
    return {"source": source, "text": text, "audio_wav_b64": _wav_b64(audio, sr), "sample_rate": sr,
            "report": rep, "note": note}


@app.websocket("/redteam/stream")
async def redteam_stream(ws: WebSocket):
    from app.services.grok import stream_grok_speech

    await ws.accept()
    det = STATE.get("detector")
    if det is None:
        await ws.send_json({"type": "error", "detail": "detector not loaded"})
        await ws.close()
        return
    try:
        start = json.loads(await ws.receive_text())
        sa, source = None, None
        async for ch in stream_grok_speech(start.get("text"), start.get("voice")):
            source = ch.source
            if sa is None:
                sa = StreamAnalyzer(det, ch.sample_rate, config.WINDOW_MS, config.HOP_MS)
            await ws.send_json({"type": "audio.chunk", "pcm16_b64": base64.b64encode(ch.pcm16).decode(),
                                "sample_rate": ch.sample_rate, "seq": ch.seq, "source": ch.source})
            sa.push(pcm_bytes_to_float(ch.pcm16, "s16le"))
            while sa.ready():
                await ws.send_json(await asyncio.to_thread(sa.step))
        if sa is None:
            raise RuntimeError("no audio produced")
        sa.flush()
        while sa.ready():
            await ws.send_json(await asyncio.to_thread(sa.step))
        rep = await asyncio.to_thread(_analyze_array, sa.audio(), file_name=f"grok-{source}")
        await ws.send_json({"type": "redteam.done", "source": source, "report": rep})
        await ws.close()
    except WebSocketDisconnect:
        return
    except Exception as e:  # noqa: BLE001
        try:
            await ws.send_json({"type": "error", "detail": f"{type(e).__name__}: {str(e)[:200]}"})
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


@app.get("/eval/summary")
def eval_summary():
    p = config.OUTPUT_DIR / "eval_summary.json"
    if not p.exists():
        raise HTTPException(404, "evaluation not computed yet")
    return JSONResponse(json.loads(p.read_text()))


class ExplainReq(BaseModel):
    report: dict


@app.post("/explain")
async def explain(req: ExplainReq):
    from app.services.grok import GrokUnavailable, explain_report

    try:
        return await explain_report(req.report)
    except GrokUnavailable as e:
        raise HTTPException(503, str(e)) from e
