"""12-Factor Cloud Configuration Loader."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration with cloud environment variable overrides."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    # 12-Factor App environment overrides for Docker / AWS EC2 / GitHub Actions
    if os.getenv("MLFLOW_TRACKING_URI"):
        cfg["model"]["mlflow_tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]
    if os.getenv("DEPLOYMENT_SLOT"):
        cfg["cloud"]["default_deployment_slot"] = os.environ["DEPLOYMENT_SLOT"]
    if os.getenv("ACTIVE_PROJECT_ID"):
        cfg["project"]["default_project_id"] = os.environ["ACTIVE_PROJECT_ID"]
    if os.getenv("PSI_THRESHOLD"):
        cfg["monitoring"]["psi_threshold"] = float(os.environ["PSI_THRESHOLD"])
    if os.getenv("KS_P_VALUE_THRESHOLD"):
        cfg["monitoring"]["ks_p_value_threshold"] = float(
            os.environ["KS_P_VALUE_THRESHOLD"]
        )
    return cfg
