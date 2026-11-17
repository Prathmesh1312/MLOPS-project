# ==============================================================================
# Stage 1: Builder — Compile wheels & install dependencies in isolated venv
# ==============================================================================
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ==============================================================================
# Stage 2: Runtime — Hardened lightweight python:3.11-slim-bookworm image
# ==============================================================================
FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="CloudMLOps Studio — Plug-and-Play CI/CD/CT Platform" \
      org.opencontainers.image.description="Multi-Project MLOps Pipeline with Centralized Registry, Dynamic & Ensemble CD Serving, and Drift CT" \
      org.opencontainers.image.base.name="python:3.11-slim-bookworm"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    SERVICE_ROLE="control-plane" \
    DEPLOYMENT_SLOT="blue" \
    ACTIVE_PROJECT_ID="house_price" \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY configs/ ./configs/
COPY src/ ./src/
COPY app/ ./app/
COPY scripts/ ./scripts/

RUN mkdir -p /app/projects /app/data/raw /app/data/processed /app/models/registry /app/logs /app/mlruns && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/ready || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
