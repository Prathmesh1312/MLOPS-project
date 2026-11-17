"""FastAPI Cloud Microservice with Observability, Drift Monitoring, and Zero-Downtime CT."""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse

from app.schemas import (
    BatchMicroClimateInput,
    CloudStatusResponse,
    DriftSimulationRequest,
    MicroClimateInput,
    PredictionResponse,
    SlotSwitchRequest,
)
from src.config import PROJECT_ROOT, load_config
from src.data.dataset import RAW_FEATURES, generate_microclimate_dataset
from src.models.registry import model_manager
from src.monitoring.drift import evaluate_data_drift
from src.monitoring.logger import telemetry_logger
from src.monitoring.webhook import trigger_continuous_training


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure Champion Ensemble model and baseline datasets are initialized on container startup."""
    model_manager.ensure_loaded()
    yield


app = FastAPI(
    title="Cloud MLOps CI/CD/CT Pipeline — Reservoir Evaporation Service",
    description=(
        "Cloud-native FastAPI microservice serving a Multi-Model Ensemble Regressor "
        "with DVC + Cloudflare R2 artifact storage, MLflow model registry, "
        "Prometheus observability, and Drift-Triggered Continuous Training (CT)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, Any]:
    """Service root endpoint with cloud runtime pointers."""
    status = model_manager.get_status()
    return {
        "message": "Cloud MLOps CI/CD/CT Pipeline API is running",
        "dashboard_url": "/dashboard",
        "docs_url": "/docs",
        "metrics_url": "/metrics",
        "deployment_slot": status["deployment_slot"],
        "container_id": status["container_id"],
        "active_model_version": status["active_model"].get("version", "v1.0"),
        "active_model_name": status["active_model"].get("model_name", "Unknown"),
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    """Serve the interactive Cloud Computing & MLOps Control Plane UI."""
    html_path = Path(__file__).resolve().parent / "static" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard UI file not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health/live")
def liveness_probe() -> dict[str, str]:
    """Container Liveness Probe (Docker / Kubernetes / AWS ALB)."""
    return {"status": "alive", "container_id": model_manager.container_id}


@app.get("/health/ready")
def readiness_probe() -> dict[str, Any]:
    """Container Readiness Probe verifying in-memory ensemble model availability."""
    status = model_manager.get_status()
    if not status["is_ready"]:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {
        "status": "ready",
        "deployment_slot": status["deployment_slot"],
        "container_id": status["container_id"],
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

    lines = [
        "# HELP mlops_requests_total Total inference requests handled by container",
        "# TYPE mlops_requests_total counter",
        f'mlops_requests_total{{slot="{status["deployment_slot"]}"}} {obs["total_requests"]}',
        "# HELP mlops_prediction_latency_avg_ms Rolling average inference latency in milliseconds",
        "# TYPE mlops_prediction_latency_avg_ms gauge",
        f"mlops_prediction_latency_avg_ms {obs['avg_latency_ms']}",
        "# HELP mlops_prediction_latency_p95_ms Rolling p95 inference latency in milliseconds",
        "# TYPE mlops_prediction_latency_p95_ms gauge",
        f"mlops_prediction_latency_p95_ms {obs['p95_latency_ms']}",
        "# HELP mlops_drift_psi_max Maximum Population Stability Index across micro-climate features",
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
        deployment_slot=status["deployment_slot"],
        is_ready=status["is_ready"],
        hot_reload_count=status["hot_reload_count"],
        active_model=status["active_model"],
        observability=obs,
        storage_backend={
            "dvc_enabled": (PROJECT_ROOT / ".dvc").exists(),
            "r2_bucket": cfg["cloud"]["r2_bucket_name"],
            "r2_endpoint_template": cfg["cloud"]["r2_endpoint_template"],
            "mlflow_tracking_uri": cfg["model"]["mlflow_tracking_uri"],
        },
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(data: MicroClimateInput) -> PredictionResponse:
    """Run multi-model ensemble inference and log production telemetry."""
    raw_dict = {col: getattr(data, col) for col in RAW_FEATURES}
    result = model_manager.predict_single(raw_dict)

    # Convert mm/day evaporation over reservoir_area_km2 into m^3/day (1 mm over 1 km^2 = 1000 m^3)
    water_loss_m3 = round(
        result["evaporation_rate_mm_day"] * data.reservoir_area_km2 * 1000.0, 2
    )
    result["estimated_water_loss_m3_day"] = water_loss_m3

    telemetry_logger.log_prediction(raw_dict, result, source="rest_api")
    return PredictionResponse(**result)


@app.post("/predict/batch")
def predict_batch(batch: BatchMicroClimateInput) -> dict[str, Any]:
    """Run batch inference across multiple micro-climate telemetry records."""
    outputs = []
    for item in batch.records:
        raw_dict = {col: getattr(item, col) for col in RAW_FEATURES}
        res = model_manager.predict_single(raw_dict)
        res["estimated_water_loss_m3_day"] = round(
            res["evaporation_rate_mm_day"] * item.reservoir_area_km2 * 1000.0, 2
        )
        telemetry_logger.log_prediction(raw_dict, res, source="batch_api")
        outputs.append(res)

    return {
        "count": len(outputs),
        "deployment_slot": model_manager.deployment_slot,
        "model_version": model_manager.get_status()["active_model"].get("version"),
        "predictions": outputs,
    }


@app.get("/logs/recent")
def get_recent_telemetry(limit: int = Query(default=25, ge=1, le=200)) -> dict[str, Any]:
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
        "recent_predictions": telemetry_logger.get_recent_logs(limit=limit),
        "ct_events": list(reversed(ct_events[-10:])),
    }


@app.post("/drift/check")
def check_drift(auto_trigger_ct: bool = Query(default=True)) -> dict[str, Any]:
    """Evaluate statistical drift (KS-test + PSI) on current production batch."""
    report = evaluate_data_drift()
    ct_event = None
    if report["overall_drift_detected"] and auto_trigger_ct:
        ct_event = trigger_continuous_training(report, execute_local_retrain=True)
    return {
        "drift_report": report,
        "continuous_training_event": ct_event,
    }


@app.post("/drift/simulate")
def simulate_drift_and_ct(req: DriftSimulationRequest) -> dict[str, Any]:
    """
    Simulate an incoming production data window (clean or drifted), log sample requests,
    evaluate statistical drift (KS-test + PSI), and automatically trigger Continuous Training
    + zero-downtime model hot-reload if drift exceeds threshold.
    """
    cfg = load_config()
    sim_df = generate_microclimate_dataset(
        n_samples=req.n_samples,
        seed=int(os.urandom(2).hex(), 16) % 10000,
        drift_mode=req.drift_mode,
        drift_intensity=req.drift_intensity,
    )

    prod_path = PROJECT_ROOT / cfg["data"]["production_batch_path"]
    sim_df.to_csv(prod_path, index=False)

    # Log a sample of incoming requests through the inference pipeline
    for _, row in sim_df.head(8).iterrows():
        raw_payload = {col: float(row[col]) for col in RAW_FEATURES}
        pred_res = model_manager.predict_single(raw_payload)
        telemetry_logger.log_prediction(
            raw_payload,
            pred_res,
            source="drift_simulator" if req.drift_mode else "normal_traffic_simulator",
        )

    drift_report = evaluate_data_drift(current_df=sim_df, config=cfg)
    ct_event = None
    if drift_report["overall_drift_detected"] and req.auto_trigger_ct:
        ct_event = trigger_continuous_training(
            drift_report,
            new_production_df=sim_df,
            execute_local_retrain=True,
        )

    return {
        "simulated_mode": "drifted_regime" if req.drift_mode else "normal_regime",
        "samples_generated": req.n_samples,
        "drift_report": drift_report,
        "continuous_training_triggered": ct_event is not None,
        "continuous_training_event": ct_event,
        "active_model_after": model_manager.get_status()["active_model"],
    }


@app.post("/cloud/reload-model")
def reload_model_endpoint() -> dict[str, Any]:
    """Zero-downtime atomic hot-reload of the Champion model artifact."""
    status = model_manager.reload_model(auto_train_if_missing=True)
    return {
        "message": "Champion model hot-reloaded with zero downtime",
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
