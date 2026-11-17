"""Tests for Simple Interpretable ML Models, Dual-Mode Serving Bundle, and Centralized Registry."""
from __future__ import annotations

from src.data.dataset import generate_project_dataset
from src.models.ensemble import build_candidate_models
from src.models.registry import model_manager
from src.models.train import load_central_registry, train_and_promote


def test_build_candidate_models_includes_simple_interpretable_models() -> None:
    models = build_candidate_models(
        {
            "dt_max_depth": 5,
            "rf_n_estimators": 20,
            "rf_max_depth": 6,
            "gb_n_estimators": 20,
            "gb_learning_rate": 0.1,
            "gb_max_depth": 3,
        }
    )
    expected = {
        "LinearRegression",
        "DecisionTree",
        "RandomForest",
        "GradientBoosting",
        "VotingEnsemble",
    }
    assert expected == set(models.keys())


def test_train_and_promote_registers_model_and_data_version() -> None:
    df = generate_project_dataset(project_id="house_price", n_samples=300, seed=77)
    meta = train_and_promote(
        trigger_reason="pytest_unit_verification",
        custom_df=df,
        project_id="house_price",
    )
    assert meta["promoted"] is True
    assert meta["metrics"]["r2"] >= 0.75
    assert len(meta["leaderboard"]) == 5
    assert "data_version" in meta
    assert meta["data_version"]["data_version"].startswith("data-v")

    registry = load_central_registry()
    assert len(registry) >= 1
    assert any(r["version"] == meta["version"] and r["project_id"] == "house_price" for r in registry)


def test_dual_mode_cd_serving_dynamic_and_ensemble() -> None:
    status = model_manager.reload_model(auto_train_if_missing=True, project_id="house_price")
    assert status["is_ready"] is True

    sample_input = {
        "area_sqft": 2100.0,
        "bedrooms": 3.0,
        "bathrooms": 2.0,
        "age_years": 10.0,
        "distance_city_km": 7.5,
    }

    res_dynamic = model_manager.predict_single(
        sample_input, project_id="house_price", serving_mode="dynamic"
    )
    assert res_dynamic["serving_mode"] == "dynamic"
    assert res_dynamic["prediction"] == res_dynamic["dynamic_output"]["prediction"]
    assert res_dynamic["dynamic_output"]["selected_model"] in (
        "LinearRegression",
        "DecisionTree",
        "RandomForest",
        "GradientBoosting",
    )

    res_ensemble = model_manager.predict_single(
        sample_input, project_id="house_price", serving_mode="ensemble"
    )
    assert res_ensemble["serving_mode"] == "ensemble"
    assert res_ensemble["prediction"] == res_ensemble["ensemble_output"]["prediction"]
    assert len(res_ensemble["ensemble_output"]["member_predictions"]) == 4
