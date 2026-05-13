# syntax=docker/dockerfile:1

# Build the React frontend inside Docker so the FastAPI image always contains
# a fresh frontend/dist bundle, even though local dist output is ignored.
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend

# Copy package metadata first to maximize npm layer caching.
COPY frontend/package*.json ./
RUN npm ci

# Build the static SPA that FastAPI serves from /app/frontend/dist.
COPY frontend/ ./
RUN npm run build


# Build Python dependencies in a separate stage so build tools do not remain
# in the runtime image.
FROM python:3.11-slim AS python-builder

WORKDIR /build

# build-essential is needed by some Python/ML packages during wheel builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install requirements after copying only requirements.txt for better caching.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt


# Runtime image: Python 3.11 slim plus only runtime OS libraries.
FROM python:3.11-slim

WORKDIR /app

# ffmpeg supports document/media parsing paths; libgomp1 supports ML runtimes.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user for safer production containers.
RUN useradd -m -u 1000 appuser

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface

COPY --from=python-builder /opt/venv /opt/venv

# Copy only the backend code and the generated frontend bundle required by
# backend/app/main.py to serve the SPA and support direct /chat refreshes.
COPY --chown=appuser:appuser backend/ ./backend/
COPY --chown=appuser:appuser --from=frontend-builder /frontend/dist ./frontend/dist

# Runtime-writable paths. docker-compose mounts backend/logs and data to keep
# logs, uploads, and caches across container restarts.
RUN mkdir -p /app/backend/logs /app/data \
    && chown -R appuser:appuser /app/backend/logs /app/data

USER appuser

EXPOSE 8000

# Local health check uses the service inside this container; inter-container
# dependencies such as Qdrant are configured through docker-compose env vars.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://127.0.0.1:8000/api/health', timeout=3).raise_for_status()"

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
