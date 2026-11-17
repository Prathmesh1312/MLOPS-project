# Cloud Computing Course Project: End-to-End CI/CD/CT MLOps Pipeline

An end-to-end **Cloud Computing & MLOps Pipeline** implementing **Continuous Integration (CI)**, **Continuous Deployment (CD)**, and **Continuous Training (CT)** for a **Multi-Model Ensemble Regressor** predicting reservoir water evaporation rates from micro-climate telemetry.

---

## 1. Cloud Computing & DevOps Architecture Highlights

| Cloud / DevOps Pillar | Implementation in Project |
| :--- | :--- |
| **Source & Data Versioning** | **GitHub** + **DVC (`dvc.yaml`)** paired with **Cloudflare R2** (S3-compatible zero-egress cloud object storage) and a local DVC storage remote (`.dvc_local_storage`) for 100% offline/local execution. |
| **Multi-Model Ensemble & Registry** | Trains `RidgeBaseline`, `RandomForest`, `XGBoost`, `LightGBM`, `VotingEnsemble`, and `StackingEnsemble` (`src/models/ensemble.py`), tracks all runs in **MLflow** (`sqlite:///mlflow.db`), and promotes the Champion model (`models/champion_ensemble.joblib`) when quality gates ($R^2 \ge 0.85$) pass. |
| **Multi-Stage Containerization** | `Dockerfile` built on `python:3.11-slim-bookworm` using a two-stage build (`builder` $\rightarrow$ `runtime`), running as a non-root user (`appuser`) with native `/health/ready` container `HEALTHCHECK` probes. |
| **Zero-Downtime Deployment** | **Infrastructure Level**: **Nginx Reverse Proxy** (`nginx/nginx.conf`) + `scripts/deploy_zero_downtime.sh` routing between `api-blue` and `api-green` containers.<br>**Application Level**: Thread-safe `CloudModelManager` supporting atomic in-memory model hot-swapping (`POST /cloud/reload-model`). |
| **Cloud Telemetry & Observability** | Logs every production inference payload, latency, model version, and container slot to `logs/production_telemetry.jsonl`, and exposes Kubernetes/Docker probes (`/health/live`, `/health/ready`) plus Prometheus metrics (`/metrics`). |
| **Continuous Training (CT) Loop** | Evaluates incoming production micro-climate distributions against baseline reference data using **Two-Sample Kolmogorov-Smirnov (KS) Test** and **Population Stability Index (PSI)**. When drift exceeds the threshold ($\text{PSI} \ge 0.25$ or $p < 0.01$), it automatically dispatches the CT webhook, retrains the ensemble, promotes the new model in MLflow, and hot-reloads with zero downtime. |

---

## 2. Quickstart (Run 100% Locally on Your Machine)

### Step 1: Activate the Local Virtual Environment
```bash
source .venv/bin/activate
```

### Step 2: Configure DVC Remotes (Local Storage + Cloudflare R2) & Run DVC Pipeline
```bash
bash scripts/setup_cloud_remotes.sh
dvc repro
dvc push
```

### Step 3: Run the Automated CI Test Suite (`pytest`)
```bash
pytest -v
```

### Step 4: Run the End-to-End Cloud CI/CD/CT Simulation Script
This single command demonstrates baseline training, MLflow promotion, liveness/readiness probes, normal traffic check, micro-climate drift injection, automated Continuous Training (CT) retraining, zero-downtime model upgrade (`v1.0` $\rightarrow$ `v2.0`), Blue-Green slot switching, and Prometheus `/metrics`:
```bash
python scripts/demo_cloud_pipeline.py
```

### Step 5: Launch the Live FastAPI Service & Interactive Cloud Control Plane UI
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Cloud & MLOps Control Plane Dashboard**: `http://localhost:8000/dashboard`
- **Interactive OpenAPI / Swagger Docs**: `http://localhost:8000/docs`
- **Prometheus Metrics Scrape Endpoint**: `http://localhost:8000/metrics`
- **Liveness & Readiness Probes**: `http://localhost:8000/health/live` & `http://localhost:8000/health/ready`

### Optional: Launch MLflow Tracking UI Locally
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### Optional: Run the Multi-Container Blue-Green Stack with Docker Compose
```bash
docker compose up --build -d
```
- **Nginx Load Balancer**: `http://localhost:8080/dashboard`
- **Blue Container Slot**: `http://localhost:8001/health/ready`
- **Green Container Slot**: `http://localhost:8002/health/ready`

---

## 3. GitHub Actions Workflows (`.github/workflows/`)
1. **[ci.yml](file:///.github/workflows/ci.yml)**: Executes data validation, trains baseline ensemble, runs `pytest`, and smoke-tests the multi-stage `python:3.11-slim-bookworm` Docker container on every push/PR.
2. **[cd.yml](file:///.github/workflows/cd.yml)**: Builds and pushes the multi-stage container image to GitHub Container Registry (`ghcr.io`) and executes zero-downtime Blue-Green deployment on AWS EC2 via `scripts/deploy_zero_downtime.sh`.
3. **[ct.yml](file:///.github/workflows/ct.yml)**: Triggered via `repository_dispatch` (`micro_climate_drift_detected`) webhook when the drift monitor detects micro-climate distribution shift; pulls the latest dataset from Cloudflare R2 via DVC, retrains the ensemble, logs to MLflow, and triggers zero-downtime deployment.
