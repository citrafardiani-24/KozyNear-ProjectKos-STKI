# ============================================================================
# KozyNear single-container deploy
# React frontend + FastAPI backend serve dari 1 service
#
# Multi-stage build:
#   Stage 1 (Node)  -> npm install + vite build -> /app/frontend/dist
#   Stage 2 (Python) -> copy frontend dist + install Python deps + uvicorn
#
# Routes di runtime:
#   GET /                  -> React SPA (index.html)
#   GET /assets/*          -> built JS/CSS chunks
#   GET /favicon.ico, dll  -> static files
#   GET /health            -> liveness probe (Render uses this)
#   GET /api/search, /api/listings/{id}, /api/eval/*  -> FastAPI endpoints
#   GET /api/docs          -> Swagger UI
# ============================================================================

# ----------------------------------------------------------------------------
# Stage 1: Build React frontend
# ----------------------------------------------------------------------------
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Layer caching: install deps dulu (gak invalidate kalau code berubah)
COPY frontend/package*.json ./
RUN npm install

# Build
COPY frontend ./
# VITE_API_URL kosong -> frontend pakai relative URL (same origin sebagai API)
ENV VITE_API_URL=""
RUN npm run build


# ----------------------------------------------------------------------------
# Stage 2: Python runtime
# ----------------------------------------------------------------------------
FROM python:3.11-slim

# System deps untuk lxml (parsing HTML scraper) + psycopg2 + faiss (libgomp1:
# wheel faiss-cpu link ke libgomp yang tidak ada di python-slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt-dev \
        libpq-dev \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install RUNTIME deps + NEURAL deps. Satu image untuk dua platform:
# - Render free 512MB: ENABLE_NEURAL=false -> fastembed/faiss tidak di-import,
#   RAM aman (paket cuma duduk di disk).
# - HF Spaces 16GB ($0): ENABLE_NEURAL=true -> MiniLM + Hybrid live.
# Full set dev (scraping, pandas, dev tools) ada di requirements.txt.
COPY backend/requirements-runtime.txt backend/requirements-neural.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-runtime.txt \
        -r requirements-neural.txt

# Pre-download model ONNX MiniLM ke cache yang dipakai runtime
# (FASTEMBED_CACHE_PATH) supaya startup dengan ENABLE_NEURAL=true tidak
# download ~120MB tiap cold start (HF Spaces disk-nya ephemeral).
ENV FASTEMBED_CACHE_PATH=/app/.fastembed_cache
RUN python -c "import os; from fastembed import TextEmbedding; \
    TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', \
    cache_dir=os.environ['FASTEMBED_CACHE_PATH'])"

# Copy backend code
COPY backend ./backend
# Copy data folder (mamikos_real_v2 source + corpus.json + pre-built indexes)
COPY data ./data
# Copy hasil evaluasi (CSV) — dibaca /api/eval/* (file-based dashboard)
COPY eval ./eval

# Copy built frontend dari stage 1
COPY --from=frontend-builder /app/frontend/dist ./static

# Working dir backend supaya alembic + relative imports berfungsi
WORKDIR /app/backend

# Render expose $PORT via env (default 10000 kalau gak set)
ENV PORT=10000
EXPOSE 10000

# Startup pipeline:
#   1. alembic upgrade head -- create/update schema (idempotent)
#   2. seed_db.py --truncate --skip-if-synced: reconcile listings table ke
#      mamikos_real_v2.jsonl (227 real, sesudah filter deskripsi kosong).
#      --skip-if-synced: kalau count + digest id DB == source, seed di-skip
#      (cold start lebih cepat); kalau beda, TRUNCATE CASCADE + reseed atomic.
#   3. uvicorn start FastAPI -- lifespan load TF-IDF + BM25 + gazetteer.
#      Neural (MiniLM) ikut di-load HANYA kalau ENABLE_NEURAL=true
#      (HF Spaces 16GB); di Render free biarkan false. Default search
#      model = "smart" (query understanding + geo + BM25).
#   Catatan resiliency: alembic/seed TIDAK boleh mematikan container kalau
#   DB unreachable (mis. Space baru yang DATABASE_URL-nya belum di-set).
#   App tetap up: /health, frontend, /api/stats (fallback corpus), /api/eval,
#   /api/preprocess jalan; /api/search yang butuh DB akan error jelas dan
#   /api/status menampilkan database.connected=false untuk debugging.
CMD (alembic upgrade head || echo "[startup] alembic SKIPPED (DB unreachable? cek DATABASE_URL; continuing)") \
    && (python -m scripts.seed_db --input ../data/raw/mamikos_real_v2.jsonl --truncate --skip-if-synced || echo "[startup] seed_db SKIPPED (continuing)") \
    && uvicorn app.main:app --host 0.0.0.0 --port $PORT
