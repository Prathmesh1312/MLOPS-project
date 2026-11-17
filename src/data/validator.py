"""Data Schema & Physical Constraints Gate for CI/CD/CT Pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data.dataset import RAW_FEATURES, TARGET_COLUMN

PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "temperature_c": (-30.0, 60.0),
    "relative_humidity_pct": (0.0, 100.0),
    "wind_speed_ms": (0.0, 45.0),
    "solar_radiation_mj": (0.0, 50.0),
    "pressure_kpa": (50.0, 115.0),
    "water_temp_c": (-5.0, 55.0),
    "evaporation_rate_mm_day": (0.0, 35.0),
}


@dataclass
class ValidationReport:
    is_valid: bool
    row_count: int
    missing_columns: list[str] = field(default_factory=list)
    null_counts: dict[str, int] = field(default_factory=dict)
    out_of_bounds_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "row_count": self.row_count,
            "missing_columns": self.missing_columns,
            "null_counts": self.null_counts,
            "out_of_bounds_counts": self.out_of_bounds_counts,
            "errors": self.errors,
        }


def validate_dataframe(
    df: pd.DataFrame, require_target: bool = True, min_rows: int = 10
) -> ValidationReport:
    """Validate dataset schema, null ratios, and physical sensor bounds."""
    errors: list[str] = []
    required = list(RAW_FEATURES) + ([TARGET_COLUMN] if require_target else [])
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

        if not np.all(np.isfinite(df[col].dropna().to_numpy(dtype=float))):
            errors.append(f"Column '{col}' contains non-finite (inf) values.")

        if col in PHYSICAL_BOUNDS:
            low, high = PHYSICAL_BOUNDS[col]
            invalid_mask = (df[col] < low) | (df[col] > high)
            n_oob = int(invalid_mask.sum())
            if n_oob > 0:
                out_of_bounds[col] = n_oob
                errors.append(
                    f"Column '{col}' has {n_oob} values outside physical range [{low}, {high}]."
                )

    return ValidationReport(
        is_valid=len(errors) == 0,
        row_count=len(df),
        missing_columns=missing_cols,
        null_counts=null_counts,
        out_of_bounds_counts=out_of_bounds,
        errors=errors,
    )
