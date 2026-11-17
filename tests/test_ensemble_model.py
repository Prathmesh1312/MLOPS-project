"""Tests for Multi-Model Ensemble Regressor, MLflow Tracking, and Champion Registry."""
from __future__ import annotations

from src.data.dataset import generate_microclimate_dataset
from src.models.ensemble import build_candidate_models
from src.models.registry import model_manager
from src.models.train import train_and_promote


def test_build_candidate_models_includes_all_regressors() -> None:
    models = build_candidate_models(
        {
            "ridge_alpha": 1.0,
            "rf_n_estimators": 20,
            "rf_max_depth": 5,
            "xgb_n_estimators": 20,
            "xgb_learning_rate": 0.1,
            "xgb_max_depth": 3,
            "lgbm_n_estimators": 20,
            "lgbm_learning_rate": 0.1,
        }
    )
    expected = {
        "RidgeBaseline",
        "RandomForest",
        "XGBoost",
        "LightGBM",
        "VotingEnsemble",
        "StackingEnsemble",
    }
    assert expected.issubset(set(models.keys()))


def test_train_and_promote_logs_to_mlflow_and_passes_gate() -> None:
    df = generate_microclimate_dataset(n_samples=350, seed=99)
    meta = train_and_promote(
        trigger_reason="pytest_unit_verification",
        custom_df=df,
    )
    assert meta["promoted"] is True
    assert meta["metrics"]["r2"] >= 0.85
    assert meta["metrics"]["rmse"] < 1.0
    assert len(meta["leaderboard"]) == 6


def test_model_manager_zero_downtime_hot_reload_and_inference() -> None:
    status_before = model_manager.reload_model(auto_train_if_missing=True)
    assert status_before["is_ready"] is True

    pred = model_manager.predict_single(
        {
            "temperature_c": 29.0,
            "relative_humidity_pct": 50.0,
            "wind_speed_ms": 3.0,
            "solar_radiation_mj": 21.0,
            "pressure_kpa": 98.5,
            "water_temp_c": 26.5,
        }
    )
    assert pred["evaporation_rate_mm_day"] > 0.0
    assert pred["inference_latency_ms"] >= 0.0
    assert pred["deployment_slot"] in ("blue", "green")
