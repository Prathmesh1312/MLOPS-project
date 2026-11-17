"""
End-to-End Local Cloud MLOps Pipeline Demonstration Script.
Demonstrates:
  1. DVC Data Preparation & Schema Validation Gate
  2. Multi-Model Ensemble Training & MLflow Registry Promotion (v1.0)
  3. FastAPI Cloud Microservice Liveness, Readiness & Inference Logging
  4. Normal Traffic Check (No Drift -> No Retraining)
  5. Micro-Climate Drift Injection -> KS-Test & PSI Threshold Breach
  6. Automated Continuous Training (CT) Webhook -> Ensemble Retraining -> Promotion (v2.0)
  7. Zero-Downtime Atomic Model Hot-Swap & Blue-Green Slot Routing
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app
from src.data.dataset import prepare_datasets
from src.models.train import train_and_promote


def print_banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def run_demo() -> None:
    print_banner("STEP 1: DATA PREPARATION & INITIAL MULTI-MODEL TRAINING (MLflow)")
    paths = prepare_datasets()
    print(f"Reference Dataset : {paths['reference_path']}")
    print(f"Processed Dataset : {paths['processed_train_path']}")

    meta_v1 = train_and_promote(trigger_reason="demo_initial_v1_build")
    print(
        f"Promoted Champion : {meta_v1['model_name']} ({meta_v1['version']}) "
        f"| R2={meta_v1['metrics']['r2']} | RMSE={meta_v1['metrics']['rmse']} mm/day"
    )
    print("MLflow Leaderboard:")
    for row in meta_v1["leaderboard"]:
        print(
            f"  - {row['model_name']:<18} | RMSE={row['rmse']:<7} | MAE={row['mae']:<7} | R2={row['r2']:<7}"
        )

    with TestClient(app) as client:
        # Ensure manager is synced with latest trained v1 artifact
        client.post("/cloud/reload-model")

        print_banner("STEP 2: CLOUD MICROSERVICE PROBES & LIVE INFERENCE TELEMETRY")
        live = client.get("/health/live").json()
        ready = client.get("/health/ready").json()
        print(f"Liveness Probe    : {live}")
        print(f"Readiness Probe   : {ready}")

        sample_req = {
            "temperature_c": 30.5,
            "relative_humidity_pct": 46.0,
            "wind_speed_ms": 3.6,
            "solar_radiation_mj": 23.8,
            "pressure_kpa": 98.5,
            "water_temp_c": 27.1,
            "reservoir_area_km2": 15.0,
        }
        pred = client.post("/predict", json=sample_req).json()
        print(
            f"Inference Output  : Evaporation = {pred['evaporation_rate_mm_day']} mm/day "
            f"(FAO-56 Baseline = {pred['fao56_baseline_mm_day']} mm/day)"
        )
        print(
            f"Cloud Metadata    : Model={pred['model_name']} ({pred['model_version']}) "
            f"| Slot={pred['deployment_slot']} | Latency={pred['inference_latency_ms']} ms"
        )

        print_banner("STEP 3: NORMAL PRODUCTION TRAFFIC (NO STATISTICAL DRIFT)")
        normal_sim = client.post(
            "/drift/simulate",
            json={
                "drift_mode": False,
                "n_samples": 300,
                "drift_intensity": 1.0,
                "auto_trigger_ct": True,
            },
        ).json()
        rep_clean = normal_sim["drift_report"]
        print(
            f"Normal Regime Check -> Drift Detected: {rep_clean['overall_drift_detected']} "
            f"| Max PSI: {rep_clean['max_psi']} | CT Triggered: {normal_sim['continuous_training_triggered']}"
        )

        print_banner("STEP 4: MICRO-CLIMATE DRIFT INJECTION -> AUTOMATED CT & ZERO-DOWNTIME SWAP")
        drift_sim = client.post(
            "/drift/simulate",
            json={
                "drift_mode": True,
                "n_samples": 350,
                "drift_intensity": 1.3,
                "auto_trigger_ct": True,
            },
        ).json()
        rep_drift = drift_sim["drift_report"]
        ct_ev = drift_sim["continuous_training_event"]
        print(
            f"Drifted Regime Check -> Drift Detected: {rep_drift['overall_drift_detected']} "
            f"| Max PSI: {rep_drift['max_psi']} | Drifted Features: {rep_drift['drifted_features']}"
        )
        print(
            f"Automated CT Result  -> Webhook Mode: {ct_ev['webhook']['mode']} "
            f"| Promoted Model: {ct_ev['promoted_model_name']} ({ct_ev['promoted_model_version']}) "
            f"| Zero-Downtime Hot-Reload: {ct_ev['zero_downtime_hot_reload']}"
        )

        print_banner("STEP 5: BLUE-GREEN SLOT SWITCH & PROMETHEUS OBSERVABILITY METRICS")
        slot_res = client.post("/cloud/switch-slot", json={"slot": "green"}).json()
        print(f"Slot Switch         : {slot_res['message']}")
        pred_after = client.post("/predict", json=sample_req).json()
        print(
            f"Post-CT Inference   : Served by {pred_after['model_name']} ({pred_after['model_version']}) "
            f"on [{pred_after['deployment_slot'].upper()}] in {pred_after['inference_latency_ms']} ms"
        )
        print("\nPrometheus /metrics scrape preview:")
        metrics_text = client.get("/metrics").text
        for line in metrics_text.strip().splitlines():
            if not line.startswith("#"):
                print(f"  {line}")


if __name__ == "__main__":
    run_demo()
