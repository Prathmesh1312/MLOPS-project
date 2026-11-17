# CloudMLOps Studio — Plug-and-Play Multi-Project CI/CD/CT Platform

A modular **MLOps & Cloud Microservices Platform** implementing **Continuous Integration (CI)**, **Dual-Mode Continuous Deployment (CD)**, and **Drift-Triggered Continuous Training (CT)** with SHA-256 dataset versioning and a Centralized Model Registry.

---

## 1. Platform Architecture

| Component | Implementation |
| :--- | :--- |
| **Interpretable ML Estimators** | Trains 4 base models (`LinearRegression`, `DecisionTree`, `RandomForest`, `GradientBoosting`) and a weighted `VotingEnsemble` ([src/models/ensemble.py](file:///Users/prathmeshtoke/Local/Project/Python/ml%20&%20mlops/ml-ci-cd-ct-pipeline/src/models/ensemble.py)). |
| **Plug-and-Play Multi-Project Workspace** | Includes 3 pre-configured projects (`house_price`, `student_score`, `cloud_cost`) and supports creating (`POST /projects/init`) or removing (`DELETE /projects/{project_id}`) custom ML projects on demand ([src/data/dataset.py](file:///Users/prathmeshtoke/Local/Project/Python/ml%20&%20mlops/ml-ci-cd-ct-pipeline/src/data/dataset.py)). |
| **Centralized Model & Data Registry** | Every training run snapshots the dataset with a **SHA-256 fingerprint** (`data-v1.0`, `data-v2.0`, ...) and registers the trained bundle (`v1.0`, `v2.0`, ...) in `models/central_registry.json` and MLflow SQLite, with 1-click zero-downtime version rollback ([src/models/train.py](file:///Users/prathmeshtoke/Local/Project/Python/ml%20&%20mlops/ml-ci-cd-ct-pipeline/src/models/train.py)). |
| **Dual-Mode CD Serving (Dynamic & Ensemble)** | **Dynamic Output**: Routes requests at runtime to the optimal individual estimator (`best_accuracy`, `fastest_latency`, or `interpretable_tree`).<br>**Ensemble Output**: Evaluates all base estimators in parallel and combines their votes via `VotingEnsemble`, returning individual sub-model predictions, 95% confidence interval, and agreement score ([src/models/registry.py](file:///Users/prathmeshtoke/Local/Project/Python/ml%20&%20mlops/ml-ci-cd-ct-pipeline/src/models/registry.py)). |
| **Live Pipeline Workflow Graph Viewer** | Interactive DAG workflow visualizer (`GET /workflow/graph`) showing connected pipeline stages (`Data Ingestion & DVC Hash` $\rightarrow$ `Schema & Bounds Gate` $\rightarrow$ `Multi-Model Training (CI)` $\rightarrow$ `Central Model Registry` $\rightarrow$ `Dual-Mode CD Serving` $\rightarrow$ `KS/PSI Drift & Auto-CT`) with per-stage durations and click-to-inspect telemetry ([app/static/index.html](file:///Users/prathmeshtoke/Local/Project/Python/ml%20&%20mlops/ml-ci-cd-ct-pipeline/app/static/index.html)). |
| **Containerized Microservices** | Packaged into 6 Docker containers in [docker-compose.yml](file:///Users/prathmeshtoke/Local/Project/Python/ml%20&%20mlops/ml-ci-cd-ct-pipeline/docker-compose.yml) (`mlops-control-plane`, `mlops-serving-blue`, `mlops-serving-green`, `mlops-drift-ct-worker`, `mlops-mlflow-server`, `mlops-nginx-gateway`). |

---

## 2. Container Services (`docker-compose.yml`)

| Container Name | Port | Role | Health Endpoint |
| :--- | :--- | :--- | :--- |
| `mlops-control-plane` | `8000` | Web UI Control Plane, Project Manager & Centralized Model Registry | `http://localhost:8000/health/ready` |
| `mlops-serving-blue` | `8001` | CD Model Serving Container (Blue Slot — Dynamic & Ensemble Output) | `http://localhost:8001/health/ready` |
| `mlops-serving-green` | `8002` | CD Model Serving Container (Green Slot — Zero-Downtime Standby/Active) | `http://localhost:8002/health/ready` |
| `mlops-drift-ct-worker` | `8003` | Statistical Drift Monitor (KS-Test + PSI) & Automated Continuous Training Worker | `http://localhost:8003/health` |
| `mlops-mlflow-server` | `5000` | MLflow Experiment Tracking & Artifact UI Server (`sqlite:///mlflow.db`) | `http://localhost:5000/` |
| `mlops-nginx-gateway` | `8080` | Nginx Reverse Proxy & Zero-Downtime Blue/Green Load Balancer | `http://localhost:8080/health/live` |

---

## 3. Quickstart

### Run Locally
```bash
source .venv/bin/activate
pytest -v
python scripts/demo_cloud_pipeline.py
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Control Plane UI**: `http://localhost:8000/dashboard`
- **OpenAPI Docs**: `http://localhost:8000/docs`
- **Prometheus Metrics**: `http://localhost:8000/metrics`

### Run Multi-Container Stack
```bash
docker compose up --build -d
docker compose ps
```
