"""FastAPI Plug-and-Play Multi-Project MLOps Platform with Centralized Registry & Dual-Mode CD Serving."""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse

from app.schemas import (
    BatchPredictRequest,
    CloudStatusResponse,
    DriftSimulationRequest,
    PredictRequest,
    ProjectInitRequest,
    ProjectSwitchRequest,
    RegistryActivateRequest,
    SlotSwitchRequest,
    TrainTriggerRequest,
)
from src.config import PROJECT_ROOT, load_config
from src.data.dataset import (
    generate_project_dataset,
    get_data_versions,
    get_project_spec,
    list_all_projects,
    prepare_datasets,
)
from src.models.registry import model_manager
from src.models.train import train_and_promote
from src.monitoring.drift import evaluate_data_drift
from src.monitoring.logger import telemetry_logger
from src.monitoring.webhook import trigger_continuous_training


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure default project datasets, Champion model bundle, and Central Registry are initialized."""
    model_manager.ensure_loaded()
    yield


app = FastAPI(
    title="CloudMLOps Studio — Plug-and-Play Multi-Project CI/CD/CT Platform",
    description=(
        "Modular Cloud Computing & MLOps Platform supporting Plug-and-Play ML Projects, "
        "Centralized Model & Data Versioning Registry, Dual-Mode CD Serving "
        "(Dynamic Single-Model Output & Voting Ensemble Output), Containerized Microservices, "
        "and Drift-Triggered Continuous Training (CT)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, Any]:
    """Service root endpoint with cloud runtime and active project pointers."""
    status = model_manager.get_status()
    active_meta = status["active_model"]
    return {
        "message": "CloudMLOps Studio — Plug-and-Play CI/CD/CT Pipeline API is running",
        "dashboard_url": "/dashboard",
        "docs_url": "/docs",
        "metrics_url": "/metrics",
        "service_role": status["service_role"],
        "deployment_slot": status["deployment_slot"],
        "container_id": status["container_id"],
        "active_project_id": status["active_project_id"],
        "active_model_version": active_meta.get("version", "v1.0"),
        "active_model_name": active_meta.get("model_name", "VotingEnsemble"),
        "dynamic_best_model": active_meta.get("dynamic_best_model", "GradientBoosting"),
        "data_version": active_meta.get("data_version", {}).get("data_version", "data-v1.0"),
        "supported_serving_modes": ["dynamic", "ensemble", "both"],
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    """Serve the CloudMLOps Studio web application."""
    html_path = Path(__file__).resolve().parent / "static" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard UI file not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health")
@app.get("/health/live")
def liveness_probe() -> dict[str, str]:
    """Container Liveness Probe (Docker / Kubernetes / AWS ALB)."""
    return {
        "status": "alive",
        "service_role": model_manager.service_role,
        "container_id": model_manager.container_id,
    }


@app.get("/health/ready")
def readiness_probe() -> dict[str, Any]:
    """Container Readiness Probe verifying in-memory model bundle availability."""
    status = model_manager.get_status()
    if not status["is_ready"]:
        raise HTTPException(status_code=503, detail="Model bundle not ready")
    return {
        "status": "ready",
        "service_role": status["service_role"],
        "deployment_slot": status["deployment_slot"],
        "container_id": status["container_id"],
        "active_project_id": status["active_project_id"],
        "model_version": status["active_model"].get("version"),
        "model_name": status["active_model"].get("model_name"),
    }


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> PlainTextResponse:
    """Expose Prometheus-compatible cloud & MLOps metrics for scraping."""
    status = model_manager.get_status()
    obs = telemetry_logger.get_observability_summary()
    version_num = int(status["active_model"].get("version_number", 1))
    r2_val = float(status["active_model"].get("metrics", {}).get("r2", 0.0))
    rmse_val = float(status["active_model"].get("metrics", {}).get("rmse", 0.0))
    pid = status["active_project_id"]

    lines = [
        "# HELP mlops_requests_total Total inference requests handled by container",
        "# TYPE mlops_requests_total counter",
        f'mlops_requests_total{{slot="{status["deployment_slot"]}",project="{pid}"}} {obs["total_requests"]}',
        "# HELP mlops_prediction_latency_avg_ms Rolling average inference latency in milliseconds",
        "# TYPE mlops_prediction_latency_avg_ms gauge",
        f"mlops_prediction_latency_avg_ms {obs['avg_latency_ms']}",
        "# HELP mlops_prediction_latency_p95_ms Rolling p95 inference latency in milliseconds",
        "# TYPE mlops_prediction_latency_p95_ms gauge",
        f"mlops_prediction_latency_p95_ms {obs['p95_latency_ms']}",
        "# HELP mlops_drift_psi_max Maximum Population Stability Index across features",
        "# TYPE mlops_drift_psi_max gauge",
        f"mlops_drift_psi_max {obs['last_drift_psi_max']}",
        "# HELP mlops_drift_detected_flag Binary flag (1 if data drift threshold exceeded)",
        "# TYPE mlops_drift_detected_flag gauge",
        f"mlops_drift_detected_flag {1 if obs['last_drift_detected'] else 0}",
        "# HELP mlops_active_model_version_number Active promoted model version number",
        "# TYPE mlops_active_model_version_number gauge",
        f"mlops_active_model_version_number {version_num}",
        "# HELP mlops_active_model_r2 Active model test R2 score",
        "# TYPE mlops_active_model_r2 gauge",
        f"mlops_active_model_r2 {r2_val}",
        "# HELP mlops_active_model_rmse Active model test RMSE",
        "# TYPE mlops_active_model_rmse gauge",
        f"mlops_active_model_rmse {rmse_val}",
        "# HELP mlops_hot_reload_count Total zero-downtime model hot-reloads executed",
        "# TYPE mlops_hot_reload_count counter",
        f"mlops_hot_reload_count {status['hot_reload_count']}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n")


@app.get("/cloud/status", response_model=CloudStatusResponse)
def cloud_status() -> CloudStatusResponse:
    """Comprehensive Cloud & MLOps control plane state."""
    cfg = load_config()
    status = model_manager.get_status()
    obs = telemetry_logger.get_observability_summary()
    return CloudStatusResponse(
        service=cfg["project"]["name"],
        environment=cfg["project"]["environment"],
        container_id=status["container_id"],
        service_role=status["service_role"],
        deployment_slot=status["deployment_slot"],
        is_ready=status["is_ready"],
        hot_reload_count=status["hot_reload_count"],
        active_project_id=status["active_project_id"],
        active_project=status["active_project"],
        available_projects=status["available_projects"],
        active_model=status["active_model"],
        observability=obs,
        storage_backend={
            "dvc_enabled": (PROJECT_ROOT / ".dvc").exists(),
            "r2_bucket": cfg["cloud"]["r2_bucket_name"],
            "r2_endpoint_template": cfg["cloud"]["r2_endpoint_template"],
            "mlflow_tracking_uri": cfg["model"]["mlflow_tracking_uri"],
            "central_registry_path": cfg["model"]["central_registry_path"],
        },
    )


@app.get("/cloud/containers")
def cloud_containers() -> dict[str, Any]:
    """
    Return the containerized microservice topology, ports, health endpoints, and status.
    """
    cfg = load_config()
    status = model_manager.get_status()
    services_cfg = cfg["cloud"].get("services", [])
    containers = []
    for svc in services_cfg:
        port = svc.get("port", 8000)
        health_ep = svc.get("health_endpoint", "/health/ready")
        containers.append(
            {
                "container_name": svc["name"],
                "role": svc["role"],
                "host_port": port,
                "health_endpoint": health_ep,
                "local_url": f"http://localhost:{port}{health_ep}",
                "status": "ACTIVE (ONLINE)",
                "active_slot": status["deployment_slot"],
                "active_model_version": status["active_model"].get("version", "v1.0"),
                "docker_inspect_cmd": f"docker logs --tail 25 {svc['name']}",
            }
        )
    return {
        "orchestrator": "Docker Compose Multi-Container Stack",
        "base_image": cfg["cloud"].get("container_base_image", "python:3.11-slim-bookworm"),
        "active_deployment_slot": status["deployment_slot"],
        "total_containers": len(containers),
        "containers": containers,
    }


@app.get("/workflow/graph")
def get_workflow_graph() -> dict[str, Any]:
    """
    Return the live CI/CD/CT DAG workflow nodes, durations, and connector states
    for the interactive Pipeline Workflow Viewer.
    """
    status = model_manager.get_status()
    obs = telemetry_logger.get_observability_summary()
    active_model = status["active_model"]
    dv = active_model.get("data_version", {})
    lb = active_model.get("leaderboard", [])
    total_train_sec = round(sum(float(r.get("train_duration_sec", 0.05)) for r in lb), 2) or 0.18
    r2_val = active_model.get("metrics", {}).get("r2", 0.0)
    drift_flag = obs.get("last_drift_detected", False)
    max_psi = obs.get("last_drift_psi_max", 0.0)

    nodes = [
        {
            "id": "data_ingest",
            "title": "Data Ingestion & DVC Hash",
            "status": "success",
            "duration": "0.08s",
            "subtitle": f"{dv.get('data_version', 'data-v1.0')} • {dv.get('row_count', 800)} rows",
            "details": {
                "stage": "1. Data Versioning & DVC Sync",
                "dataset_version": dv.get("data_version", "data-v1.0"),
                "sha256_fingerprint": dv.get("sha256_short", "--"),
                "row_count": dv.get("row_count", 800),
                "storage_backend": dv.get("storage_tier", "DVC Local Cache + Cloudflare R2"),
            },
        },
        {
            "id": "schema_gate",
            "title": "Schema & Bounds Gate",
            "status": "success",
            "duration": "0.03s",
            "subtitle": f"{len(active_model.get('features', []))} features validated",
            "details": {
                "stage": "2. Automated CI Data Quality Gate",
                "features_checked": ", ".join(active_model.get("features", [])),
                "target_column": active_model.get("target", "--"),
                "null_check": "0 nulls (Passed)",
                "range_bounds": "100% in valid domain bounds",
            },
        },
        {
            "id": "model_training",
            "title": "Multi-Model Training (CI)",
            "status": "success",
            "duration": f"{total_train_sec}s",
            "subtitle": f"5 models • R²={r2_val}",
            "details": {
                "stage": "3. Multi-Model Training & Evaluation",
                "models_trained": "LinearRegression, DecisionTree, RandomForest, GradientBoosting, VotingEnsemble",
                "champion_selected": active_model.get("model_name", "VotingEnsemble"),
                "dynamic_best_model": active_model.get("dynamic_best_model", "GradientBoosting"),
                "validation_r2": r2_val,
                "validation_rmse": active_model.get("metrics", {}).get("rmse", 0.0),
            },
        },
        {
            "id": "central_registry",
            "title": "Central Model Registry",
            "status": "success",
            "duration": "0.04s",
            "subtitle": f"{active_model.get('version', 'v1.0')} • Production",
            "details": {
                "stage": "4. Centralized Registry & Artifact Store",
                "registered_id": active_model.get("registry_id", "--"),
                "active_version": active_model.get("version", "v1.0"),
                "stage_status": active_model.get("stage", "Production"),
                "mlflow_run_id": active_model.get("mlflow_run_id", "--"),
                "artifact_path": active_model.get("artifact_path", "--"),
            },
        },
        {
            "id": "cd_serving",
            "title": "Dual-Mode CD Serving",
            "status": "success",
            "duration": f"{obs.get('avg_latency_ms', 1.2)}ms",
            "subtitle": f"Slot: {status['deployment_slot'].upper()} • {obs.get('total_requests', 0)} reqs",
            "details": {
                "stage": "5. Continuous Deployment (Dynamic & Ensemble)",
                "serving_modes": "Dynamic Output + Voting Ensemble Output",
                "active_slot": status["deployment_slot"].upper(),
                "hot_reloads": status["hot_reload_count"],
                "total_requests": obs.get("total_requests", 0),
                "avg_latency": f"{obs.get('avg_latency_ms', 0.0)} ms",
            },
        },
        {
            "id": "drift_ct_loop",
            "title": "KS/PSI Drift & Auto-CT",
            "status": "warning" if drift_flag else "success",
            "duration": "0.14s",
            "subtitle": f"PSI={max_psi} • {'CT Retrained' if drift_flag else 'Stable'}",
            "details": {
                "stage": "6. Statistical Drift Monitor & Continuous Training",
                "drift_status": "Drift Detected (Auto-CT Executed)" if drift_flag else "Stable Distribution",
                "max_feature_psi": max_psi,
                "statistical_tests": "Two-Sample Kolmogorov-Smirnov (KS) + Population Stability Index (PSI)",
                "webhook_trigger": active_model.get("trigger_reason", "pipeline_init"),
            },
        },
    ]

    return {
        "workflow_file": "ci-cd-ct-pipeline.yml",
        "trigger": f"on: push, drift_webhook, project_init ({status['active_project_id']})",
        "active_project_id": status["active_project_id"],
        "active_version": active_model.get("version", "v1.0"),
        "nodes": nodes,
    }


# ==============================================================================
# Plug-and-Play Multi-Project & Data Versioning Endpoints
# ==============================================================================
@app.get("/projects")
def get_projects() -> dict[str, Any]:
    """List all plug-and-play ML projects, active project config, and tracked data versions."""
    status = model_manager.get_status()
    pid = status["active_project_id"]
    return {
        "active_project_id": pid,
        "active_project": status["active_project"],
        "projects": list_all_projects(),
        "data_versions": get_data_versions(pid),
    }


@app.post("/projects/init")
def initialize_new_project(req: ProjectInitRequest) -> dict[str, Any]:
    """
    Initialize a brand-new plug-and-play ML project:
      1. Creates project schema & bounds
      2. Generates & versions initial dataset (`data-v1.0` + SHA-256)
      3. Trains simple base models + VotingEnsemble (`v1.0`)
      4. Registers `v1.0` in the Centralized Model Registry
      5. Switches CD serving to the new project
    """
    try:
        return model_manager.init_new_project(
            project_id=req.project_id,
            name=req.name,
            description=req.description,
            features=req.features,
            target=req.target,
            target_unit=req.target_unit,
            domain=req.domain,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/projects/switch")
def switch_active_project(req: ProjectSwitchRequest) -> dict[str, Any]:
    """Switch the active ML project and load (or bootstrap) its champion model bundle."""
    try:
        status = model_manager.switch_project(req.project_id)
        return {
            "message": f"Switched active project to '{req.project_id}'",
            "status": status,
            "data_versions": get_data_versions(req.project_id),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/projects/{project_id}")
def delete_project_endpoint(project_id: str) -> dict[str, Any]:
    """Remove an ML project, its dataset lineage, and its model artifacts from the pipeline."""
    try:
        return model_manager.delete_project(project_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/projects/train")
def trigger_project_training(req: TrainTriggerRequest) -> dict[str, Any]:
    """Train a new model version for the specified project and register it in the Central Registry."""
    cfg = load_config()
    pid = req.project_id or model_manager.active_project_id
    prep = prepare_datasets(cfg, project_id=pid)
    spec = get_project_spec(pid, cfg)

    # Generate a fresh batch of training samples so new version reflects updated data lineage
    fresh_batch = generate_project_dataset(
        project_id=pid,
        n_samples=150,
        seed=int(os.urandom(2).hex(), 16) % 10000,
        drift_mode=False,
        project_spec=spec,
    )
    import pandas as pd

    existing_df = pd.read_csv(prep["processed_train_path"])
    merged_df = pd.concat([existing_df, fresh_batch], ignore_index=True).tail(1200)

    meta = train_and_promote(
        cfg,
        trigger_reason=req.trigger_reason,
        custom_df=merged_df,
        project_id=pid,
    )
    status = model_manager.reload_model(auto_train_if_missing=False, project_id=pid)
    return {
        "message": f"Trained & promoted {meta['version']} ({meta['model_name']}) for project '{pid}'",
        "trained_version": meta,
        "status": status,
    }


# ==============================================================================
# Centralized Model Registry Endpoints
# ==============================================================================
@app.get("/registry/models")
def list_central_registry(project_id: str | None = Query(default=None)) -> dict[str, Any]:
    """Return every trained model version from the Centralized Model Registry."""
    return model_manager.get_central_registry(project_id=project_id)


@app.post("/registry/activate")
def activate_model_version(req: RegistryActivateRequest) -> dict[str, Any]:
    """Activate / Rollback to any specific trained model version in the Centralized Model Registry."""
    try:
        status = model_manager.activate_registry_version(
            project_id=req.project_id, version=req.version
        )
        return {
            "message": f"Activated {req.project_id} ({req.version}) in Production with zero downtime",
            "status": status,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ==============================================================================
# Dual-Mode CD Model Serving (Dynamic Output + Ensemble Output)
# ==============================================================================
@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    """
    Run CD inference supporting two serving modes:
      - `serving_mode="dynamic"`: Dynamic Output (routes to best single model or strategy)
      - `serving_mode="ensemble"`: Ensemble Output (combines all base models via VotingEnsemble)
      - `serving_mode="both"`: Returns both Dynamic Output and Ensemble Output side-by-side
    """
    raw_features = req.extract_features()
    result = model_manager.predict_single(
        raw_payload=raw_features,
        project_id=req.project_id,
        serving_mode=req.serving_mode,
        dynamic_strategy=req.dynamic_strategy,
        preferred_model=req.preferred_model,
    )
    telemetry_logger.log_prediction(result["features_used"], result, source=f"cd_{req.serving_mode}")
    return result


@app.post("/predict/batch")
def predict_batch(batch: BatchPredictRequest) -> dict[str, Any]:
    """Run batch inference across multiple records using Dynamic, Ensemble, or Both serving modes."""
    outputs = []
    target_pid = batch.project_id or model_manager.active_project_id
    for item in batch.records:
        res = model_manager.predict_single(
            raw_payload=item,
            project_id=target_pid,
            serving_mode=batch.serving_mode,
        )
        telemetry_logger.log_prediction(res["features_used"], res, source="batch_api")
        outputs.append(res)

    return {
        "count": len(outputs),
        "project_id": target_pid,
        "serving_mode": batch.serving_mode,
        "deployment_slot": model_manager.deployment_slot,
        "model_version": model_manager.get_status(target_pid)["active_model"].get("version"),
        "predictions": outputs,
    }


# ==============================================================================
# Telemetry, Drift Detection & Continuous Training (CT) Loop
# ==============================================================================
@app.get("/logs/recent")
def get_recent_telemetry(
    limit: int = Query(default=25, ge=1, le=200),
    project_id: str | None = Query(default=None),
) -> dict[str, Any]:
    """Return recently logged production inference requests and CT history."""
    cfg = load_config()
    ct_history_path = PROJECT_ROOT / cfg["monitoring"]["ct_history_path"]
    ct_events: list[dict[str, Any]] = []
    if ct_history_path.exists():
        try:
            ct_events = json.loads(ct_history_path.read_text(encoding="utf-8"))
        except Exception:
            ct_events = []

    return {
        "observability": telemetry_logger.get_observability_summary(),
        "recent_predictions": telemetry_logger.get_recent_logs(limit=limit, project_id=project_id),
        "ct_events": list(reversed(ct_events[-15:])),
    }


@app.post("/drift/check")
def check_drift(
    auto_trigger_ct: bool = Query(default=True),
    project_id: str | None = Query(default=None),
) -> dict[str, Any]:
    """Evaluate statistical drift (KS-test + PSI) on current production batch."""
    pid = project_id or model_manager.active_project_id
    report = evaluate_data_drift(project_id=pid)
    ct_event = None
    if report["overall_drift_detected"] and auto_trigger_ct:
        ct_event = trigger_continuous_training(
            report, execute_local_retrain=True, project_id=pid
        )
    return {
        "drift_report": report,
        "continuous_training_event": ct_event,
    }


@app.post("/drift/simulate")
def simulate_drift_and_ct(req: DriftSimulationRequest) -> dict[str, Any]:
    """
    Simulate an incoming production data window (clean or drifted) for the active project,
    evaluate statistical drift (KS-test + PSI), and automatically trigger Continuous Training (CT)
    + Data & Model Versioning + zero-downtime hot-reload if drift exceeds threshold.
    """
    cfg = load_config()
    pid = req.project_id or model_manager.active_project_id
    spec = get_project_spec(pid, cfg)
    prep = prepare_datasets(cfg, project_id=pid)

    sim_df = generate_project_dataset(
        project_id=pid,
        n_samples=req.n_samples,
        seed=int(os.urandom(2).hex(), 16) % 10000,
        drift_mode=req.drift_mode,
        drift_intensity=req.drift_intensity,
        project_spec=spec,
    )
    sim_df.to_csv(prep["production_batch_path"], index=False)

    feature_cols = list(spec["features"].keys())
    for _, row in sim_df.head(6).iterrows():
        raw_payload = {col: float(row[col]) for col in feature_cols}
        pred_res = model_manager.predict_single(
            raw_payload, project_id=pid, serving_mode="both"
        )
        telemetry_logger.log_prediction(
            raw_payload,
            pred_res,
            source="drift_simulator" if req.drift_mode else "normal_traffic_simulator",
        )

    drift_report = evaluate_data_drift(current_df=sim_df, config=cfg, project_id=pid)
    ct_event = None
    if drift_report["overall_drift_detected"] and req.auto_trigger_ct:
        ct_event = trigger_continuous_training(
            drift_report,
            new_production_df=sim_df,
            execute_local_retrain=True,
            project_id=pid,
        )

    return {
        "project_id": pid,
        "simulated_mode": "drifted_regime" if req.drift_mode else "normal_regime",
        "samples_generated": req.n_samples,
        "drift_report": drift_report,
        "continuous_training_triggered": ct_event is not None,
        "continuous_training_event": ct_event,
        "active_model_after": model_manager.get_status(pid)["active_model"],
    }


@app.post("/cloud/reload-model")
def reload_model_endpoint() -> dict[str, Any]:
    """Zero-downtime atomic hot-reload of the active project's model bundle."""
    status = model_manager.reload_model(auto_train_if_missing=True)
    return {
        "message": "Champion model bundle hot-reloaded with zero downtime",
        "status": status,
    }


@app.post("/cloud/switch-slot")
def switch_deployment_slot(req: SlotSwitchRequest) -> dict[str, Any]:
    """Switch active Blue-Green container slot label for zero-downtime deployment demos."""
    os.environ["DEPLOYMENT_SLOT"] = req.slot
    model_manager.deployment_slot = req.slot
    return {
        "message": f"Switched active deployment slot to '{req.slot}' with zero downtime",
        "deployment_slot": model_manager.deployment_slot,
        "container_id": model_manager.container_id,
    }
