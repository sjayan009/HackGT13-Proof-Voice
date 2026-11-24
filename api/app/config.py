"""Runtime configuration (env vars, loaded from api/.env when present; never logs secrets)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]
ROOT = API_DIR.parent
if str(ROOT) not in sys.path:  # top-level `engine/` package (Aegis)
    sys.path.append(str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(API_DIR / ".env", override=False)
except Exception:  # noqa: BLE001
    pass


def _path(name: str, default: str) -> Path:
    v = os.getenv(name, default)
    p = Path(v)
    return p if p.is_absolute() else (API_DIR / p).resolve()


VERSION = "2.0.0"
MODEL_DIR = _path("PROOFVOICE_MODEL_DIR", "../models/selected")
OUTPUT_DIR = _path("OUTPUT_DIR", "../outputs")
TMP_DIR = _path("TMP_DIR", "../tmp")
DEVICE = os.getenv("DEVICE", "auto")
SR = int(os.getenv("MODEL_INPUT_SAMPLE_RATE", "16000"))
WINDOW_MS = int(os.getenv("FORENSIC_WINDOW_MS", "3000"))  # >= held-out minimum clip length (3.0 s)
HOP_MS = int(os.getenv("FORENSIC_HOP_MS", "500"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_ANALYZE_S = float(os.getenv("MAX_ANALYZE_S", "120"))
DATA_DIR = _path("DATA_DIR", "../data")

# v2 Guardian / Aegis (secrets come from api/.env only; never logged)
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_PUBLISHABLE_KEY = os.getenv("SUPABASE_PUBLISHABLE_KEY", "")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "") or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")  # e.g. https://abcd.ngrok-free.app
GUARDIAN_RECIPIENT_NUMBER = os.getenv("GUARDIAN_RECIPIENT_NUMBER", "")  # dev-mode fallback line
GUARDIAN_SMS_ALERTS = os.getenv("GUARDIAN_SMS_ALERTS", "1") == "1"
AEGIS_POLICY = os.getenv("AEGIS_POLICY", "adaptive")
AEGIS_BACKEND = os.getenv("AEGIS_BACKEND", "pytorch")  # pytorch | tensorrt (falls back to pytorch)
AEGIS_MAX_BATCH = int(os.getenv("AEGIS_MAX_BATCH", "16"))
AEGIS_MAX_WAIT_MS = float(os.getenv("AEGIS_MAX_WAIT_MS", "5"))
AEGIS_DEADLINE_MS = float(os.getenv("AEGIS_DEADLINE_MS", "1000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")


def resolve_device() -> str:
    if DEVICE != "auto":
        return DEVICE
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"
