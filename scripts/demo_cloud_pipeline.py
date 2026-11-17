"""
End-to-End Local Cloud MLOps Platform Demonstration Script.
Demonstrates:
  1. Plug-and-Play Project Initialization (`house_price`, `student_score`, `cloud_cost`, + custom project)
  2. Data Versioning (SHA-256 + DVC) & Simple Interpretable ML Models Training (`v1.0`)
  3. Centralized Model Registry Inspection across projects
  4. Dual-Mode CD Model Serving:
     - Mode 1: Dynamic Output (routes to top-performing single model or strategy)
     - Mode 2: Ensemble Output (VotingEnsemble consensus + individual base model breakdown)
  5. Statistical Drift Detection (KS-Test + PSI) -> Automated Continuous Training (CT) -> `data-v2.0` + `v2.0`
  6. Centralized Model Registry Version Rollback/Activation & Containerized Microservice Topology
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
    print("\n" + "=" * 82)
    print(f"  {title}")
    print("=" * 82)


def run_demo() -> None:
    print_banner("STEP 1: INITIALIZE BUILT-IN PLUG-AND-PLAY PROJECTS & DATA VERSIONING")
    for pid in ("house_price", "student_score", "cloud_cost"):
        prep = prepare_datasets(project_id=pid)
        meta = train_and_promote(project_id=pid, trigger_reason="initial_project_bootstrap")
        dv = meta["data_version"]
        print(
            f"  [Project: {pid:<13}] Model: {meta['version']} ({meta['model_name']}) "
            f"| Dynamic Best: {meta['dynamic_best_model']} | R2={meta['metrics']['r2']} "
            f"| Data: {dv['data_version']} (SHA256={dv['sha256_short']})"
        )

    with TestClient(app) as client:
        client.post("/projects/switch", json={"project_id": "house_price"})

        print_banner("STEP 2: PLUG-AND-PLAY INIT OF A BRAND NEW USER ML PROJECT ('car_price')")
        init_res = client.post(
            "/projects/init",
            json={
                "project_id": "car_price",
                "name": "Used Car Resale Value Predictor",
                "description": "Predict car price from horsepower, mileage, age, and fuel economy.",
                "features": ["horsepower", "mileage_km_k", "age_years", "km_per_liter"],
                "target": "resale_price_usd_k",
                "target_unit": "$1,000 USD",
            },
        ).json()
        v1_car = init_res["initial_model_version"]
        print(f"  Initialized Project : {init_res['project']['name']} ({init_res['project']['project_id']})")
        print(
            f"  Tracked Version     : {v1_car['version']} | Data Version: {v1_car['data_version']['data_version']} "
            f"| Champion: {v1_car['model_name']} (R2={v1_car['metrics']['r2']})"
        )

        # Switch back to default house_price project
        client.post("/projects/switch", json={"project_id": "house_price"})

        print_banner("STEP 3: DUAL-MODE CD SERVING — DYNAMIC OUTPUT VS. ENSEMBLE OUTPUT")
        house_input = {
            "project_id": "house_price",
            "serving_mode": "both",
            "dynamic_strategy": "best_accuracy",
            "features": {
                "area_sqft": 2200.0,
                "bedrooms": 4.0,
                "bathrooms": 3.0,
                "age_years": 8.0,
                "distance_city_km": 6.5,
            },
        }
        pred = client.post("/predict", json=house_input).json()
        dyn = pred["dynamic_output"]
        ens = pred["ensemble_output"]
        print(
            f"  [1] Dynamic Output  : {dyn['prediction']} {pred['target_unit']} "
            f"| Selected Model: {dyn['selected_model']} ({dyn['selection_reason']})"
        )
        print(
            f"  [2] Ensemble Output : {ens['prediction']} {pred['target_unit']} "
            f"| Agreement: {ens['agreement_score_pct']}% | 95% CI: {ens['confidence_interval_95']}"
        )
        print(f"      Individual Base Model Votes: {ens['member_predictions']}")

        print_banner("STEP 4: DRIFT INJECTION -> AUTOMATED CONTINUOUS TRAINING (CT) & VERSIONING")
        drift_sim = client.post(
            "/drift/simulate",
            json={
                "project_id": "house_price",
                "drift_mode": True,
                "n_samples": 260,
                "drift_intensity": 1.35,
                "auto_trigger_ct": True,
            },
        ).json()
        rep_drift = drift_sim["drift_report"]
        ct_ev = drift_sim["continuous_training_event"]
        print(
            f"  Drift Check         : Drift Detected={rep_drift['overall_drift_detected']} "
            f"| Max PSI={rep_drift['max_psi']} | Drifted Features={rep_drift['drifted_features']}"
        )
        print(
            f"  Automated CT Event  : Promoted {ct_ev['promoted_model_name']} ({ct_ev['promoted_model_version']}) "
            f"trained on {ct_ev['data_version']} | Zero-Downtime Hot-Reload={ct_ev['zero_downtime_hot_reload']}"
        )

        print_banner("STEP 5: CENTRALIZED MODEL REGISTRY & CONTAINER TOPOLOGY FOR PRESENTATION")
        reg = client.get("/registry/models").json()
        print(f"  Total Versions in Centralized Registry: {reg['total_versions']}")
        for entry in reg["versions"][:6]:
            dv_info = entry.get("data_version", {})
            print(
                f"    - [{entry['project_id']:<13}] {entry['version']} ({entry['stage']:<10}) "
                f"| Model={entry['model_name']:<16} | R2={entry['metrics']['r2']:<6} "
                f"| Data={dv_info.get('data_version')} ({dv_info.get('sha256_short')})"
            )

        containers = client.get("/cloud/containers").json()
        print(f"\n  Containerized Services ({containers['total_containers']} containers):")
        for c in containers["containers"]:
            print(f"    - {c['container_name']:<24} | Port {c['host_port']:<5} | {c['role']}")


if __name__ == "__main__":
    run_demo()
