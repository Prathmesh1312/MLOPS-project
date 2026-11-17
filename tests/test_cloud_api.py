"""Integration tests for FastAPI Cloud Microservice endpoints, probes, metrics, and CT loop."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_cloud_health_probes_and_prometheus_metrics() -> None:
    with TestClient(app) as client:
        r_root = client.get("/")
        assert r_root.status_code == 200
        assert "active_model_version" in r_root.json()

        r_live = client.get("/health/live")
        assert r_live.status_code == 200
        assert r_live.json()["status"] == "alive"

        r_ready = client.get("/health/ready")
        assert r_ready.status_code == 200
        assert r_ready.json()["status"] == "ready"

        r_metrics = client.get("/metrics")
        assert r_metrics.status_code == 200
        assert "mlops_requests_total" in r_metrics.text
        assert "mlops_active_model_r2" in r_metrics.text

        r_dash = client.get("/dashboard")
        assert r_dash.status_code == 200
        assert "Cloud MLOps CI/CD/CT Control Plane" in r_dash.text


def test_single_and_batch_prediction_endpoints() -> None:
    with TestClient(app) as client:
        payload = {
            "temperature_c": 30.2,
            "relative_humidity_pct": 48.0,
            "wind_speed_ms": 3.4,
            "solar_radiation_mj": 23.1,
            "pressure_kpa": 98.7,
            "water_temp_c": 26.8,
            "reservoir_area_km2": 10.0,
        }
        r_pred = client.post("/predict", json=payload)
        assert r_pred.status_code == 200
        body = r_pred.json()
        assert body["evaporation_rate_mm_day"] > 0
        assert body["estimated_water_loss_m3_day"] > 0
        assert body["deployment_slot"] in ("blue", "green")

        r_batch = client.post("/predict/batch", json={"records": [payload, payload]})
        assert r_batch.status_code == 200
        assert r_batch.json()["count"] == 2


def test_slot_switch_hot_reload_and_drift_simulation() -> None:
    with TestClient(app) as client:
        r_slot = client.post("/cloud/switch-slot", json={"slot": "green"})
        assert r_slot.status_code == 200
        assert r_slot.json()["deployment_slot"] == "green"

        r_reload = client.post("/cloud/reload-model")
        assert r_reload.status_code == 200
        assert r_reload.json()["status"]["is_ready"] is True

        # Simulate drifted batch & verify CT triggers
        r_sim = client.post(
            "/drift/simulate",
            json={
                "drift_mode": True,
                "n_samples": 180,
                "drift_intensity": 1.3,
                "auto_trigger_ct": True,
            },
        )
        assert r_sim.status_code == 200
        sim_data = r_sim.json()
        assert sim_data["drift_report"]["overall_drift_detected"] is True
        assert sim_data["continuous_training_triggered"] is True
        assert sim_data["continuous_training_event"]["zero_downtime_hot_reload"] is True

        # Switch back to blue slot
        client.post("/cloud/switch-slot", json={"slot": "blue"})
