"""Micro-climate dataset generation, CSV ingestion, and compact FAO-56 feature helper."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config

RAW_FEATURES = [
    "temperature_c",
    "relative_humidity_pct",
    "wind_speed_ms",
    "solar_radiation_mj",
    "pressure_kpa",
    "water_temp_c",
]

ENGINEERED_FEATURES = [
    "vapor_pressure_deficit_kpa",
    "fao56_baseline_mm_day",
    "thermal_gradient_c",
]

ALL_MODEL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES
TARGET_COLUMN = "evaporation_rate_mm_day"


def compute_fao56_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append compact FAO-56 Penman-Monteith features to a micro-climate DataFrame."""
    out = df.copy()
    temp = out["temperature_c"].astype(float)
    rh = np.clip(out["relative_humidity_pct"].astype(float), 1.0, 100.0)
    wind = np.maximum(out["wind_speed_ms"].astype(float), 0.0)
    rad = np.maximum(out["solar_radiation_mj"].astype(float), 0.0)
    pressure = out["pressure_kpa"].astype(float)
    water_temp = out["water_temp_c"].astype(float)

    # Saturation & actual vapor pressure (kPa)
    es = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
    ea = es * (rh / 100.0)
    vpd = np.maximum(es - ea, 0.0)

    # Slope of saturation vapor pressure curve (delta) & psychrometric constant (gamma)
    delta = (4098.0 * es) / np.square(temp + 237.3)
    gamma = 0.000665 * pressure

    # Net radiation approximation for open water (albedo ~ 0.06)
    rn = 0.77 * rad
    # FAO-56 Penman-Monteith open-water baseline (mm/day)
    numerator = 0.408 * delta * rn + gamma * (900.0 / (temp + 273.0)) * wind * vpd
    denominator = delta + gamma * (1.0 + 0.34 * wind)
    fao56_mm = np.maximum(numerator / denominator, 0.0)

    out["vapor_pressure_deficit_kpa"] = np.round(vpd, 4)
    out["fao56_baseline_mm_day"] = np.round(fao56_mm, 4)
    out["thermal_gradient_c"] = np.round(water_temp - temp, 4)
    return out


def compute_single_record_features(payload: dict[str, float]) -> dict[str, float]:
    """Compute engineered features for a single telemetry payload dictionary."""
    df = pd.DataFrame([payload])
    enriched = compute_fao56_features(df)
    return {col: float(enriched.iloc[0][col]) for col in ALL_MODEL_FEATURES}


def generate_microclimate_dataset(
    n_samples: int = 1500,
    seed: int = 42,
    drift_mode: bool = False,
    drift_intensity: float = 1.0,
) -> pd.DataFrame:
    """
    Generate realistic reservoir micro-climate telemetry and target evaporation rate.
    When drift_mode=True, introduces a micro-climate regime shift (e.g., heatwave + high wind + low RH)
    to trigger the Continuous Training (CT) drift detector.
    """
    rng = np.random.default_rng(seed)
    day_of_year = rng.integers(1, 366, size=n_samples)
    seasonal_wave = np.sin(2 * np.pi * (day_of_year - 80) / 365.0)

    temp_shift = 5.5 * drift_intensity if drift_mode else 0.0
    rh_shift = -18.0 * drift_intensity if drift_mode else 0.0
    wind_shift = 1.8 * drift_intensity if drift_mode else 0.0
    rad_shift = 4.2 * drift_intensity if drift_mode else 0.0

    temperature_c = np.clip(
        24.0 + 8.5 * seasonal_wave + rng.normal(temp_shift, 2.8, n_samples),
        2.0,
        50.0,
    )
    relative_humidity_pct = np.clip(
        58.0 - 14.0 * seasonal_wave + rng.normal(rh_shift, 9.5, n_samples),
        8.0,
        99.0,
    )
    wind_speed_ms = np.clip(
        rng.gamma(shape=2.6, scale=1.1, size=n_samples) + wind_shift,
        0.2,
        18.0,
    )
    solar_radiation_mj = np.clip(
        18.5 + 6.5 * seasonal_wave + rng.normal(rad_shift, 3.0, n_samples),
        2.0,
        35.0,
    )
    pressure_kpa = np.clip(rng.normal(98.5, 1.1, size=n_samples), 85.0, 105.0)
    water_temp_c = np.clip(
        temperature_c * 0.86 + 3.2 + rng.normal(0.0, 1.1, n_samples),
        4.0,
        42.0,
    )

    df = pd.DataFrame(
        {
            "temperature_c": np.round(temperature_c, 3),
            "relative_humidity_pct": np.round(relative_humidity_pct, 3),
            "wind_speed_ms": np.round(wind_speed_ms, 3),
            "solar_radiation_mj": np.round(solar_radiation_mj, 3),
            "pressure_kpa": np.round(pressure_kpa, 3),
            "water_temp_c": np.round(water_temp_c, 3),
        }
    )

    df = compute_fao56_features(df)

    # Non-linear reservoir target combining FAO-56 baseline + thermal storage + advection interaction
    non_linear_interaction = (
        0.18 * df["thermal_gradient_c"] * np.sqrt(df["wind_speed_ms"])
        + 0.32 * np.power(df["vapor_pressure_deficit_kpa"], 1.15)
        + 0.014 * df["solar_radiation_mj"] * np.log1p(df["wind_speed_ms"])
    )
    regime_adjustment = (
        0.45 * drift_intensity * (df["temperature_c"] > 30.0).astype(float)
        if drift_mode
        else 0.0
    )
    noise = rng.normal(0.0, 0.18, size=n_samples)

    evap = np.maximum(
        0.92 * df["fao56_baseline_mm_day"]
        + non_linear_interaction
        + regime_adjustment
        + noise,
        0.1,
    )
    df[TARGET_COLUMN] = np.round(evap, 4)
    return df


def prepare_datasets(config: dict[str, Any] | None = None) -> dict[str, Path]:
    """Ensure baseline reference and processed datasets exist on disk for DVC and training."""
    cfg = config or load_config()
    ref_path = PROJECT_ROOT / cfg["data"]["reference_path"]
    prod_path = PROJECT_ROOT / cfg["data"]["production_batch_path"]
    train_path = PROJECT_ROOT / cfg["data"]["processed_train_path"]

    ref_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = int(cfg["data"].get("n_reference_samples", 1500))
    seed = int(cfg["data"].get("random_seed", 42))

    if not ref_path.exists():
        ref_df = generate_microclimate_dataset(
            n_samples=n_samples, seed=seed, drift_mode=False
        )
        ref_df.to_csv(ref_path, index=False)
    else:
        ref_df = compute_fao56_features(pd.read_csv(ref_path))
        ref_df.to_csv(ref_path, index=False)

    if not prod_path.exists():
        prod_df = generate_microclimate_dataset(
            n_samples=300, seed=seed + 10, drift_mode=False
        )
        prod_df.to_csv(prod_path, index=False)
    else:
        prod_df = compute_fao56_features(pd.read_csv(prod_path))

    combined_df = pd.concat([ref_df, prod_df], ignore_index=True)
    combined_df.to_csv(train_path, index=False)

    return {
        "reference_path": ref_path,
        "production_batch_path": prod_path,
        "processed_train_path": train_path,
    }


if __name__ == "__main__":
    paths = prepare_datasets()
    print(f"Prepared datasets: {paths}")
