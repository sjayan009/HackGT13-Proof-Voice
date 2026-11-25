"""ProofVoice voice-message checker backend on Modal: serverless, scales to zero when idle.

Public mode only (PROOFVOICE_PUBLIC=1): /health and /analyze/file. Guardian, telephony and the engine
API are not started; uploads are rate-limited per client IP. Audio is deleted after each request.

    .venv/Scripts/modal deploy deploy/modal_app.py     # prints https://<workspace>--proofvoice-check-web.modal.run
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
# Modal needs a payment method on the account for GPUs (free credits still apply). With one, set GPU = "T4":
# checks drop from ~10 s to ~1-2 s.
GPU = None

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("torch==2.6.0", index_url=None if GPU else "https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .env({"PROOFVOICE_PUBLIC": "1", "DEVICE": "cuda" if GPU else "cpu", "OMP_NUM_THREADS": "8",
          "FORENSIC_HOP_MS": "500" if GPU else "1000", "MAX_ANALYZE_S": "60", "MAX_UPLOAD_MB": "20",
          "ANALYZE_CONCURRENCY": "2", "RATE_LIMIT_PER_10MIN": "10", "HF_HUB_OFFLINE": "1",
          "TRANSFORMERS_OFFLINE": "1", "FFMPEG_PATH": "ffmpeg", "PROOFVOICE_MODEL_DIR": "/app/models/selected",
          "CORS_ORIGINS": "*"})
    .add_local_dir(ROOT / "api", "/app/api", ignore=[".env", "**/.env", "**/__pycache__", "**/tests"])
    .add_local_dir(ROOT / "engine", "/app/engine", ignore=["**/__pycache__"])
    .add_local_dir(ROOT / "models" / "selected", "/app/models/selected",
                   ignore=["trt", "*.npy", "val_report*", "train_log.json"])
)

app = modal.App("proofvoice-check", image=image)


# ponytail: one container at most caps spend (8 CPU ~$0.40/h, T4 ~$0.60/h, only while awake); raise
# max_containers if checks start to queue.
@app.function(gpu=GPU, cpu=8.0, memory=6144, max_containers=1, scaledown_window=120, timeout=300)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    import os
    import sys

    os.chdir("/app/api")
    sys.path.insert(0, "/app/api")
    from app.main import app as fastapi_app

    return fastapi_app
