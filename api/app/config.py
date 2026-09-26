"""Runtime configuration (env vars, loaded from api/.env when present; never logs secrets)."""
from __future__ import annotations

import os
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]
ROOT = API_DIR.parent

try:
    from dotenv import load_dotenv

    load_dotenv(API_DIR / ".env", override=False)
except Exception:  # noqa: BLE001
    pass


def _path(name: str, default: str) -> Path:
    v = os.getenv(name, default)
    p = Path(v)
    return p if p.is_absolute() else (API_DIR / p).resolve()


VERSION = "0.1.0"
MODEL_DIR = _path("PROOFVOICE_MODEL_DIR", "../models/selected")
OUTPUT_DIR = _path("OUTPUT_DIR", "../outputs")
TMP_DIR = _path("TMP_DIR", "../tmp")
DEVICE = os.getenv("DEVICE", "auto")
SR = int(os.getenv("MODEL_INPUT_SAMPLE_RATE", "16000"))
WINDOW_MS = int(os.getenv("FORENSIC_WINDOW_MS", "2000"))
HOP_MS = int(os.getenv("FORENSIC_HOP_MS", "500"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_ANALYZE_S = float(os.getenv("MAX_ANALYZE_S", "120"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")


def resolve_device() -> str:
    if DEVICE != "auto":
        return DEVICE
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"
