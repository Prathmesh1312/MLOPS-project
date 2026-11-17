"""Tests for Cloud Telemetry Logging, KS/PSI Statistical Drift Engine, and CT Trigger."""
from __future__ import annotations

from src.data.dataset import generate_microclimate_dataset
from src.monitoring.drift import calculate_psi, evaluate_data_drift
from src.monitoring.logger import telemetry_logger


def test_psi_is_near_zero_for_identical_distributions_and_high_for_shifted() -> None:
    ref = generate_microclimate_dataset(n_samples=500, seed=10)["temperature_c"].to_numpy()
    same = generate_microclimate_dataset(n_samples=400, seed=11)["temperature_c"].to_numpy()
    shifted = ref + 7.5

    psi_same = calculate_psi(ref, same)
    psi_shifted = calculate_psi(ref, shifted)

    assert psi_same < 0.15
    assert psi_shifted > 0.50


def test_evaluate_data_drift_distinguishes_clean_vs_drifted_batches() -> None:
    ref_df = generate_microclimate_dataset(n_samples=600, seed=42, drift_mode=False)
    clean_df = generate_microclimate_dataset(n_samples=300, seed=55, drift_mode=False)
    drifted_df = generate_microclimate_dataset(
        n_samples=300, seed=66, drift_mode=True, drift_intensity=1.25
    )

    clean_report = evaluate_data_drift(current_df=clean_df, reference_df=ref_df)
    assert clean_report["overall_drift_detected"] is False

    drift_report = evaluate_data_drift(current_df=drifted_df, reference_df=ref_df)
    assert drift_report["overall_drift_detected"] is True
    assert drift_report["max_psi"] >= 0.25
    assert len(drift_report["drifted_features"]) >= 2


def test_telemetry_logger_records_predictions_and_observability_summary() -> None:
    raw = {
        "temperature_c": 27.0,
        "relative_humidity_pct": 55.0,
        "wind_speed_ms": 2.8,
        "solar_radiation_mj": 20.0,
        "pressure_kpa": 98.5,
        "water_temp_c": 25.0,
    }
    fake_result = {
        "evaporation_rate_mm_day": 5.42,
        "fao56_baseline_mm_day": 4.95,
        "vapor_pressure_deficit_kpa": 1.60,
        "model_name": "StackingEnsemble",
        "model_version": "v1.0",
        "deployment_slot": "blue",
        "container_id": "test-node",
        "inference_latency_ms": 4.2,
    }
    rec = telemetry_logger.log_prediction(raw, fake_result, source="pytest")
    assert rec["evaporation_rate_mm_day"] == 5.42

    obs = telemetry_logger.get_observability_summary()
    assert obs["total_requests"] >= 1
    assert obs["avg_latency_ms"] > 0.0
