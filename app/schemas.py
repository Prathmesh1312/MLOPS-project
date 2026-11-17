"""Pydantic v2 Request & Response Schemas for Plug-and-Play Multi-Project MLOps Platform."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """
    Flexible inference request supporting both:
      1. Structured `{ "project_id": "...", "serving_mode": "dynamic|ensemble|both", "features": {...} }`
      2. Flat feature keys at top level (e.g. `{"area_sqft": 2100, "bedrooms": 3, ...}`)
    """

    model_config = ConfigDict(extra="allow")

    project_id: str | None = Field(
        default=None, description="Target project ID (defaults to active project)"
    )
    serving_mode: str = Field(
        default="both",
        pattern="^(dynamic|ensemble|both)$",
        description="CD Serving Mode: 'dynamic' (best single model), 'ensemble' (VotingEnsemble), or 'both'",
    )
    dynamic_strategy: str = Field(
        default="best_accuracy",
        description="Dynamic selection rule: 'best_accuracy', 'fastest_latency', or 'interpretable_tree'",
    )
    preferred_model: str | None = Field(
        default=None,
        description="Optional explicit base model name for dynamic mode",
    )
    features: dict[str, float] | None = Field(
        default=None,
        description="Feature key-value dictionary for the project",
    )

    def extract_features(self) -> dict[str, float]:
        if self.features:
            return {k: float(v) for k, v in self.features.items() if v is not None}
        reserved = {
            "project_id",
            "serving_mode",
            "dynamic_strategy",
            "preferred_model",
            "features",
        }
        extra_dict = self.model_dump()
        out: dict[str, float] = {}
        for k, v in extra_dict.items():
            if k not in reserved and isinstance(v, (int, float)):
                out[k] = float(v)
        return out


class BatchPredictRequest(BaseModel):
    project_id: str | None = None
    serving_mode: str = Field(default="both", pattern="^(dynamic|ensemble|both)$")
    records: list[dict[str, float]] = Field(..., min_length=1, max_length=500)


class ProjectInitRequest(BaseModel):
    project_id: str = Field(..., min_length=2, max_length=40, description="Unique slug e.g. 'car_price'")
    name: str = Field(..., min_length=2, max_length=80, description="Human-readable project name")
    description: str = Field(
        default="Custom Plug-and-Play ML Regression Pipeline",
        description="Project summary",
    )
    domain: str = Field(default="Custom ML Project")
    features: list[str] = Field(
        ...,
        min_length=2,
        max_length=15,
        description="List of numeric input feature names",
    )
    target: str = Field(..., min_length=2, max_length=40, description="Target column name")
    target_unit: str = Field(default="units", description="Unit of target variable")


class ProjectSwitchRequest(BaseModel):
    project_id: str = Field(..., min_length=2, max_length=50)


class TrainTriggerRequest(BaseModel):
    project_id: str | None = None
    trigger_reason: str = Field(default="manual_registry_training")


class RegistryActivateRequest(BaseModel):
    project_id: str = Field(..., min_length=2, max_length=50)
    version: str = Field(..., min_length=2, max_length=20, description="Model version tag e.g. 'v1.0'")


class DriftSimulationRequest(BaseModel):
    project_id: str | None = None
    drift_mode: bool = Field(
        default=True,
        description="If True, injects a drifted production batch to trigger Continuous Training (CT)",
    )
    n_samples: int = Field(default=250, ge=50, le=2000)
    drift_intensity: float = Field(default=1.25, ge=0.1, le=3.0)
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
    service_role: str
    deployment_slot: str
    is_ready: bool
    hot_reload_count: int
    active_project_id: str
    active_project: dict[str, Any]
    available_projects: list[dict[str, Any]]
    active_model: dict[str, Any]
    observability: dict[str, Any]
    storage_backend: dict[str, Any]
