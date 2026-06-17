# ── Stage 1: build dependencies ───────────────────────────────────────────────
# Use a slim Python base to keep the image small.
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build-time system deps (gcc required for psycopg2-binary on slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="AI Insight Agent Team"
LABEL description="Health-insurance enrollment anomaly-detection service v1"

# Non-root user for security (OWASP: Least Privilege)
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source — exclude dev artefacts via .dockerignore
COPY app/      ./app/
COPY ml/       ./ml/
COPY run.py    ./run.py

# Pre-trained model artifacts are baked in; override via volume mount in prod.
# To rebuild artifacts at container start instead, override CMD:
#   CMD ["python", "train_v1.py", "--model-dir", "/app/ml/models"]
COPY train_v1.py ./train_v1.py

# Environment defaults — all secrets MUST be supplied at runtime via env vars.
# Never hardcode credentials here.
ENV ENV=production \
    LOG_LEVEL=INFO \
    API_HOST=0.0.0.0 \
    API_PORT=8001 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8001

# Drop to non-root
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/api/v1/health')" || exit 1

# Production: single worker, no --reload
CMD ["python", "-m", "uvicorn", "app.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "1", \
     "--no-access-log"]
