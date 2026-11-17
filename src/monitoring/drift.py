"""Plug-and-Play Statistical Data Drift Engine (Kolmogorov-Smirnov + PSI)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import PROJECT_ROOT, load_config
from src.data.dataset import DEFAULT_PROJECT_ID, get_project_spec, prepare_datasets
from src.monitoring.logger import telemetry_logger


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) between reference and current arrays."""
    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(expected, quantiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    exp_pct = np.clip(exp_counts / max(len(expected), 1), 1e-4, None)
    act_pct = np.clip(act_counts / max(len(actual), 1), 1e-4, None)

    psi_value = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(max(psi_value, 0.0), 4)


def evaluate_data_drift(
    current_df: pd.DataFrame | None = None,
    reference_df: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    """
    Compare incoming production feature distribution against reference baseline dataset
    using Two-Sample Kolmogorov-Smirnov (KS) test and Population Stability Index (PSI).
    """
    cfg = config or load_config()
    pid = project_id or cfg.get("project", {}).get("default_project_id", DEFAULT_PROJECT_ID)
    spec = get_project_spec(pid, cfg)
    prep = prepare_datasets(cfg, project_id=pid)

    if reference_df is None:
        reference_df = pd.read_csv(prep["reference_path"])

    if current_df is None:
        current_df = pd.read_csv(prep["production_batch_path"])

    ks_alpha = float(cfg["monitoring"].get("ks_p_value_threshold", 0.01))
    psi_limit = float(cfg["monitoring"].get("psi_threshold", 0.20))
    min_ratio = float(cfg["monitoring"].get("min_drifted_features_ratio", 0.30))

    feature_cols = list(spec["features"].keys())
    feature_reports: dict[str, Any] = {}
    drifted_features: list[str] = []
    max_psi = 0.0

    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref_vals = reference_df[col].dropna().to_numpy(dtype=float)
        cur_vals = current_df[col].dropna().to_numpy(dtype=float)

        ks_stat, p_val = ks_2samp(ref_vals, cur_vals)
        psi_val = calculate_psi(ref_vals, cur_vals)
        max_psi = max(max_psi, psi_val)

        is_feature_drifted = bool((psi_val >= psi_limit) or (p_val < ks_alpha and psi_val >= 0.12))
        if is_feature_drifted:
            drifted_features.append(col)

        feature_reports[col] = {
            "label": spec["features"][col].get("label", col),
            "reference_mean": round(float(np.mean(ref_vals)), 3),
            "current_mean": round(float(np.mean(cur_vals)), 3),
            "mean_shift": round(float(np.mean(cur_vals) - np.mean(ref_vals)), 3),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_p_value": float(f"{p_val:.6e}"),
            "psi": psi_val,
            "drift_detected": is_feature_drifted,
        }

    total_checked = max(len(feature_reports), 1)
    drift_share = round(len(drifted_features) / total_checked, 4)
    overall_drift = drift_share >= min_ratio

    report = {
        "project_id": pid,
        "project_name": spec["name"],
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_samples": len(reference_df),
        "current_samples": len(current_df),
        "thresholds": {
            "psi_threshold": psi_limit,
            "ks_p_value_threshold": ks_alpha,
            "min_drifted_features_ratio": min_ratio,
        },
        "overall_drift_detected": overall_drift,
        "drifted_features_count": len(drifted_features),
        "drifted_features_ratio": drift_share,
        "drifted_features": drifted_features,
        "max_psi": round(max_psi, 4),
        "feature_breakdown": feature_reports,
    }

    telemetry_logger.last_drift_psi_max = round(max_psi, 4)
    telemetry_logger.last_drift_detected = overall_drift

    report_path = PROJECT_ROOT / cfg["monitoring"]["drift_report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
