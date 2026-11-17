"""Integration tests for Plug-and-Play Multi-Project MLOps API, Workflow Graph, Central Registry, Dual-Mode CD & CT."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_cloud_health_probes_workflow_graph_and_light_dashboard() -> None:
    with TestClient(app) as client:
        r_root = client.get("/")
        assert r_root.status_code == 200
        assert "active_model_version" in r_root.json()
        assert "supported_serving_modes" in r_root.json()

        r_live = client.get("/health/live")
        assert r_live.status_code == 200
        assert r_live.json()["status"] == "alive"

        r_ready = client.get("/health/ready")
        assert r_ready.status_code == 200
        assert r_ready.json()["status"] == "ready"

        r_wf = client.get("/workflow/graph")
        assert r_wf.status_code == 200
        wf_body = r_wf.json()
        assert wf_body["workflow_file"] == "ci-cd-ct-pipeline.yml"
        assert len(wf_body["nodes"]) == 6

        r_containers = client.get("/cloud/containers")
        assert r_containers.status_code == 200
        assert r_containers.json()["total_containers"] == 6

        r_metrics = client.get("/metrics")
        assert r_metrics.status_code == 200
        assert "mlops_requests_total" in r_metrics.text

        r_dash = client.get("/dashboard")
        assert r_dash.status_code == 200
        assert "CloudMLOps Studio" in r_dash.text
        assert "Faculty Presentation" not in r_dash.text


def test_plug_and_play_project_init_switch_serving_and_delete() -> None:
    with TestClient(app) as client:
        # 1. Initialize a custom project
        r_init = client.post(
            "/projects/init",
            json={
                "project_id": "crop_yield_demo",
                "name": "Crop Yield Predictor",
                "description": "Predict crop yield from rainfall, fertilizer, and soil quality",
                "features": ["rainfall_mm", "fertilizer_kg", "soil_quality_idx"],
                "target": "yield_tons_ha",
                "target_unit": "tons/ha",
            },
        )
        assert r_init.status_code == 200
        init_body = r_init.json()
        assert init_body["project"]["project_id"] == "crop_yield_demo"
        assert init_body["initial_model_version"]["version"].startswith("v")

        # 2. Test Dynamic Output Mode
        r_dyn = client.post(
            "/predict",
            json={
                "project_id": "crop_yield_demo",
                "serving_mode": "dynamic",
                "dynamic_strategy": "fastest_latency",
                "features": {"rainfall_mm": 60.0, "fertilizer_kg": 45.0, "soil_quality_idx": 70.0},
            },
        )
        assert r_dyn.status_code == 200
        dyn_data = r_dyn.json()
        assert dyn_data["serving_mode"] == "dynamic"
        assert dyn_data["dynamic_output"]["selected_model"] == "LinearRegression"

        # 3. Test Ensemble Output Mode
        r_ens = client.post(
            "/predict",
            json={
                "project_id": "crop_yield_demo",
                "serving_mode": "ensemble",
                "features": {"rainfall_mm": 60.0, "fertilizer_kg": 45.0, "soil_quality_idx": 70.0},
            },
        )
        assert r_ens.status_code == 200
        ens_data = r_ens.json()
        assert ens_data["serving_mode"] == "ensemble"
        assert "member_predictions" in ens_data["ensemble_output"]

        # 4. Delete the custom project and verify active project automatically switches
        r_del = client.delete("/projects/crop_yield_demo")
        assert r_del.status_code == 200
        del_body = r_del.json()
        assert del_body["deleted_project_id"] == "crop_yield_demo"
        assert del_body["active_project_id"] != "crop_yield_demo"

        # 5. Ensure house_price is active
        r_sw = client.post("/projects/switch", json={"project_id": "house_price"})
        assert r_sw.status_code == 200
        assert r_sw.json()["status"]["active_project_id"] == "house_price"


def test_central_registry_drift_ct_and_version_rollback() -> None:
    with TestClient(app) as client:
        # Simulate drift to trigger CT and create a new version in Centralized Registry
        r_sim = client.post(
            "/drift/simulate",
            json={
                "project_id": "house_price",
                "drift_mode": True,
                "n_samples": 180,
                "drift_intensity": 1.35,
                "auto_trigger_ct": True,
            },
        )
        assert r_sim.status_code == 200
        sim_data = r_sim.json()
        assert sim_data["drift_report"]["overall_drift_detected"] is True
        assert sim_data["continuous_training_triggered"] is True
        assert sim_data["continuous_training_event"]["zero_downtime_hot_reload"] is True

        # Query Centralized Model Registry
        r_reg = client.get("/registry/models?project_id=house_price")
        assert r_reg.status_code == 200
        reg_data = r_reg.json()
        assert reg_data["total_versions"] >= 1
        first_ver = reg_data["versions"][-1]["version"]

        # Activate / rollback to v1.0
        r_act = client.post(
            "/registry/activate",
            json={"project_id": "house_price", "version": first_ver},
        )
        assert r_act.status_code == 200
        assert r_act.json()["status"]["active_model"]["version"] == first_ver
