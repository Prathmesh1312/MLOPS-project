"""Pydantic v2 Request & Response Schemas with Cloud Runtime Metadata."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MicroClimateInput(BaseModel):
    temperature_c: float = Field(
        default=28.5, ge=-30.0, le=60.0, description="Air temperature in Celsius"
    )
    relative_humidity_pct: float = Field(
        default=52.0, ge=1.0, le=100.0, description="Relative humidity (%)"
    )
    wind_speed_ms: float = Field(
        default=3.2, ge=0.0, le=45.0, description="Wind speed at 2m height (m/s)"
    )
    solar_radiation_mj: float = Field(
        default=22.4, ge=0.0, le=50.0, description="Solar radiation (MJ/m^2/day)"
    )
    pressure_kpa: float = Field(
        default=98.6, ge=50.0, le=115.0, description="Atmospheric pressure (kPa)"
    )
    water_temp_c: float = Field(
        default=26.0, ge=-5.0, le=55.0, description="Reservoir surface water temperature (C)"
    )
    reservoir_area_km2: float = Field(
        default=12.5, ge=0.01, le=10000.0, description="Reservoir surface area (km^2)"
    )


class BatchMicroClimateInput(BaseModel):
    records: list[MicroClimateInput] = Field(..., min_length=1, max_length=500)


class PredictionResponse(BaseModel):
    evaporation_rate_mm_day: float
    fao56_baseline_mm_day: float
    vapor_pressure_deficit_kpa: float
    estimated_water_loss_m3_day: float
    model_name: str
    model_version: str
    mlflow_run_id: str
    deployment_slot: str
    container_id: str
    inference_latency_ms: float


class DriftSimulationRequest(BaseModel):
    drift_mode: bool = Field(
        default=True,
        description="If True, injects a drifted micro-climate batch to trigger CT",
    )
    n_samples: int = Field(default=300, ge=50, le=2000)
    drift_intensity: float = Field(default=1.2, ge=0.1, le=3.0)
    auto_trigger_ct: bool = Field(
        default=True,
        description="Automatically trigger Continuous Training & zero-downtime hot-swap when drift exceeds threshold",
    )


class SlotSwitchRequest(BaseModel):
    slot: str = Field(..., pattern="^(blue|green)$")


class CloudStatusResponse(BaseModel):
    service: str
    environment: str
    container_id: str
    deployment_slot: str
    is_ready: bool
    hot_reload_count: int
    active_model: dict[str, Any]
    observability: dict[str, Any]
    storage_backend: dict[str, Any]
