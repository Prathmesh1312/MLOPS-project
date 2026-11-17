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

LABEL org.opencontainers.image.title="Cloud MLOps CI/CD/CT Reservoir Evaporation API" \
      org.opencontainers.image.description="Multi-model ensemble microservice with DVC, R2, MLflow & CT" \
      org.opencontainers.image.base.name="python:3.11-slim-bookworm"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    DEPLOYMENT_SLOT="blue" \
    PORT=8000

# Install libgomp1 (required by XGBoost/LightGBM OpenMP) and curl (for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy pre-built virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source, configuration, and initial data/model directories
COPY configs/ ./configs/
COPY src/ ./src/
COPY app/ ./app/
COPY scripts/ ./scripts/

RUN mkdir -p /app/data/raw /app/data/processed /app/models /app/logs /app/mlruns && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/ready || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
