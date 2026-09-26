# ProofVoice — offline HEARSAY inference + API.  No secrets are baked in; xAI is optional at runtime.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    FFMPEG_PATH=ffmpeg DEVICE=auto \
    PROOFVOICE_MODEL_DIR=/app/models/selected OUTPUT_DIR=/app/outputs

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU torch keeps the image portable; on a CUDA host swap the index URL for cu124 wheels.
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY ml/ ml/
COPY models/selected/ models/selected/
COPY outputs/results/ outputs/results/
COPY outputs/eval_summary.json* outputs/
COPY README.md RESULTS.md* ./

# Default: generate the held-out TSV from a mounted test set.
#   docker run --rm -v "<abs>/data/hearsay:/data" -v "<abs>/outputs:/out" proofvoice
CMD ["python", "ml/generate_hearsay_tsv.py", \
     "--input", "/data/test", \
     "--template", "/data/template/HearsayScoreKey4TeamX.tsv", \
     "--output", "/out/team_predictions.tsv"]
