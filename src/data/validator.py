"""Plug-and-Play Data Schema & Bounds Validation Gate for CI/CD/CT Pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data.dataset import DEFAULT_PROJECT_ID, get_project_spec


@dataclass
class ValidationReport:
    is_valid: bool
    row_count: int
    project_id: str = DEFAULT_PROJECT_ID
    missing_columns: list[str] = field(default_factory=list)
    null_counts: dict[str, int] = field(default_factory=dict)
    out_of_bounds_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "row_count": self.row_count,
            "project_id": self.project_id,
            "missing_columns": self.missing_columns,
            "null_counts": self.null_counts,
            "out_of_bounds_counts": self.out_of_bounds_counts,
            "errors": self.errors,
        }


def validate_dataframe(
    df: pd.DataFrame,
    require_target: bool = True,
    min_rows: int = 10,
    project_id: str = DEFAULT_PROJECT_ID,
    project_spec: dict[str, Any] | None = None,
) -> ValidationReport:
    """Validate dataset schema, null ratios, and feature domain bounds for any project."""
    spec = project_spec or get_project_spec(project_id)
    feature_specs: dict[str, dict[str, Any]] = spec["features"]
    target_col: str = spec["target"]

    errors: list[str] = []
    required = list(feature_specs.keys()) + ([target_col] if require_target else [])
    missing_cols = [c for c in required if c not in df.columns]

    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if len(df) < min_rows:
        errors.append(f"Row count {len(df)} is below minimum threshold ({min_rows}).")

    null_counts: dict[str, int] = {}
    out_of_bounds: dict[str, int] = {}

    for col in required:
        if col not in df.columns:
            continue
        n_null = int(df[col].isna().sum())
        if n_null > 0:
            null_counts[col] = n_null
            errors.append(f"Column '{col}' contains {n_null} null/NaN values.")

        col_vals = df[col].dropna().to_numpy(dtype=float)
        if not np.all(np.isfinite(col_vals)):
            errors.append(f"Column '{col}' contains non-finite (inf) values.")

        if col in feature_specs:
            low = float(feature_specs[col].get("min", -1e9))
            high = float(feature_specs[col].get("max", 1e9))
            invalid_mask = (df[col] < low) | (df[col] > high)
            n_oob = int(invalid_mask.sum())
            if n_oob > 0:
                out_of_bounds[col] = n_oob
                errors.append(
                    f"Column '{col}' has {n_oob} values outside valid range [{low}, {high}]."
                )

    return ValidationReport(
        is_valid=len(errors) == 0,
        row_count=len(df),
        project_id=project_id,
        missing_columns=missing_cols,
        null_counts=null_counts,
        out_of_bounds_counts=out_of_bounds,
        errors=errors,
    )
