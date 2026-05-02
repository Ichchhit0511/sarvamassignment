# Universal Dockerfile — works on Hugging Face Spaces, Fly.io, Railway,
# Cloud Run, any container host.
#
# Single image runs both the FastAPI backend and serves the frontend.
# One URL, one process. No Netlify split needed for a demo.

FROM python:3.11-slim

# System deps for pdfplumber (poppler) + ChromaDB (sqlite headers).
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache layer for deps.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code.
COPY backend ./backend
COPY frontend ./frontend
COPY scripts ./scripts

# Hugging Face Spaces convention — they expose port 7860 by default.
# Other hosts pass $PORT. Default to 7860 for HF, fall back to 8000 locally.
ENV PORT=7860
EXPOSE 7860

# Make data dir writeable (HF Spaces / Render run as non-root).
RUN mkdir -p /app/data/manuals /app/data/chroma_db && chmod -R 777 /app/data

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
