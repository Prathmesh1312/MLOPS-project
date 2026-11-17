"""Tests for Multi-Project Cloud Telemetry Logging and KS/PSI Statistical Drift Engine."""
from __future__ import annotations

from src.data.dataset import generate_project_dataset
from src.monitoring.drift import calculate_psi, evaluate_data_drift
from src.monitoring.logger import telemetry_logger


def test_psi_is_near_zero_for_identical_distributions_and_high_for_shifted() -> None:
    ref = generate_project_dataset(project_id="house_price", n_samples=500, seed=10)["area_sqft"].to_numpy()
    same = generate_project_dataset(project_id="house_price", n_samples=400, seed=11)["area_sqft"].to_numpy()
    shifted = ref + 900.0

    psi_same = calculate_psi(ref, same)
    psi_shifted = calculate_psi(ref, shifted)

    assert psi_same < 0.15
    assert psi_shifted > 0.50


def test_evaluate_data_drift_distinguishes_clean_vs_drifted_batches() -> None:
    ref_df = generate_project_dataset(
        project_id="house_price", n_samples=500, seed=42, drift_mode=False
    )
    clean_df = generate_project_dataset(
        project_id="house_price", n_samples=250, seed=55, drift_mode=False
    )
    drifted_df = generate_project_dataset(
        project_id="house_price", n_samples=250, seed=66, drift_mode=True, drift_intensity=1.3
    )

    clean_report = evaluate_data_drift(
        current_df=clean_df, reference_df=ref_df, project_id="house_price"
    )
    assert clean_report["overall_drift_detected"] is False

    drift_report = evaluate_data_drift(
        current_df=drifted_df, reference_df=ref_df, project_id="house_price"
    )
    assert drift_report["overall_drift_detected"] is True
    assert drift_report["max_psi"] >= 0.20
    assert len(drift_report["drifted_features"]) >= 2


def test_telemetry_logger_records_predictions_and_observability_summary() -> None:
    raw = {
        "area_sqft": 1900.0,
        "bedrooms": 3.0,
        "bathrooms": 2.0,
        "age_years": 12.0,
        "distance_city_km": 9.0,
    }
    fake_result = {
        "project_id": "house_price",
        "serving_mode": "both",
        "prediction": 345.2,
        "dynamic_output": {"prediction": 342.1, "selected_model": "GradientBoosting"},
        "ensemble_output": {"prediction": 345.2},
        "model_name": "VotingEnsemble",
        "model_version": "v1.0",
        "data_version": "data-v1.0",
        "deployment_slot": "blue",
        "container_id": "test-node",
        "inference_latency_ms": 2.4,
    }
    rec = telemetry_logger.log_prediction(raw, fake_result, source="pytest")
    assert rec["prediction"] == 345.2
    assert rec["dynamic_prediction"] == 342.1
    assert rec["ensemble_prediction"] == 345.2

    obs = telemetry_logger.get_observability_summary()
    assert obs["total_requests"] >= 1
    assert obs["avg_latency_ms"] > 0.0
