"""Unit tests for dataset generation, compact FAO-56 feature engineering, and CI schema validation."""
from __future__ import annotations

import pandas as pd

from src.data.dataset import (
    ALL_MODEL_FEATURES,
    RAW_FEATURES,
    TARGET_COLUMN,
    compute_fao56_features,
    generate_microclimate_dataset,
)
from src.data.validator import validate_dataframe


def test_generate_microclimate_dataset_and_features() -> None:
    df = generate_microclimate_dataset(n_samples=120, seed=101, drift_mode=False)
    assert len(df) == 120
    for col in ALL_MODEL_FEATURES + [TARGET_COLUMN]:
        assert col in df.columns
    assert (df["fao56_baseline_mm_day"] >= 0.0).all()
    assert (df["vapor_pressure_deficit_kpa"] >= 0.0).all()
    assert (df[TARGET_COLUMN] > 0.0).all()


def test_schema_validation_passes_on_valid_data() -> None:
    df = generate_microclimate_dataset(n_samples=60, seed=42)
    report = validate_dataframe(df, require_target=True)
    assert report.is_valid is True
    assert report.row_count == 60
    assert report.errors == []


def test_schema_validation_catches_out_of_bounds_and_missing_columns() -> None:
    df = generate_microclimate_dataset(n_samples=30, seed=42)
    df.loc[0, "relative_humidity_pct"] = 145.0  # Invalid RH > 100%
    df_missing = df.drop(columns=["wind_speed_ms"])

    report = validate_dataframe(df_missing, require_target=True)
    assert report.is_valid is False
    assert "wind_speed_ms" in report.missing_columns
    assert report.out_of_bounds_counts.get("relative_humidity_pct") == 1


def test_fao56_monotonic_with_temperature_and_solar_radiation() -> None:
    cool = pd.DataFrame(
        [
            {
                "temperature_c": 18.0,
                "relative_humidity_pct": 60.0,
                "wind_speed_ms": 2.5,
                "solar_radiation_mj": 14.0,
                "pressure_kpa": 99.0,
                "water_temp_c": 17.0,
            }
        ]
    )
    hot = pd.DataFrame(
        [
            {
                "temperature_c": 36.0,
                "relative_humidity_pct": 35.0,
                "wind_speed_ms": 4.5,
                "solar_radiation_mj": 27.0,
                "pressure_kpa": 99.0,
                "water_temp_c": 31.0,
            }
        ]
    )
    cool_out = compute_fao56_features(cool)
    hot_out = compute_fao56_features(hot)
    assert (
        hot_out.iloc[0]["fao56_baseline_mm_day"]
        > cool_out.iloc[0]["fao56_baseline_mm_day"]
    )
    assert (
        hot_out.iloc[0]["vapor_pressure_deficit_kpa"]
        > cool_out.iloc[0]["vapor_pressure_deficit_kpa"]
    )
    assert set(RAW_FEATURES).issubset(set(hot_out.columns))
