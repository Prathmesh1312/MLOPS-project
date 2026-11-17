"""Plug-and-Play Multi-Project Model Training, MLflow Tracking & Centralized Model Registry."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROJECT_ROOT, load_config
from src.data.dataset import (
    DEFAULT_PROJECT_ID,
    get_project_dir,
    get_project_spec,
    prepare_datasets,
    record_data_version,
)
from src.data.validator import validate_dataframe
from src.models.ensemble import (
    BASE_MODEL_NAMES,
    ENSEMBLE_MODEL_NAME,
    MODEL_DESCRIPTIONS,
    PlugAndPlayModelBundle,
    build_candidate_models,
    evaluate_predictions,
)


def get_central_registry_path(config: dict[str, Any] | None = None) -> Path:
    cfg = config or load_config()
    rel = cfg.get("model", {}).get("central_registry_path", "models/central_registry.json")
    path = PROJECT_ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_central_registry(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Load all registered model versions across all projects from the Centralized Model Registry."""
    path = get_central_registry_path(config)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_central_registry(entries: list[dict[str, Any]], config: dict[str, Any] | None = None) -> None:
    path = get_central_registry_path(config)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def train_and_promote(
    config: dict[str, Any] | None = None,
    trigger_reason: str = "manual_or_ci_pipeline",
    custom_df: pd.DataFrame | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    """
    Train all simple candidate models + VotingEnsemble for a project:
      1. Validate schema & bounds
      2. Version the training dataset (SHA-256 + DVC metadata)
      3. Fit LinearRegression, DecisionTree, RandomForest, GradientBoosting & VotingEnsemble
      4. Log runs to MLflow and build leaderboard
      5. Save PlugAndPlayModelBundle supporting Dynamic Output & Ensemble Output
      6. Append version entry to Centralized Model Registry (`models/central_registry.json`)
    """
    cfg = config or load_config()
    pid = project_id or cfg.get("project", {}).get("default_project_id", DEFAULT_PROJECT_ID)
    spec = get_project_spec(pid, cfg)
    pdir = get_project_dir(pid, cfg)

    prep_info = prepare_datasets(cfg, project_id=pid)
    train_csv_path = prep_info["processed_train_path"]

    if custom_df is not None:
        df = custom_df.copy()
        df.to_csv(train_csv_path, index=False)
    else:
        df = pd.read_csv(train_csv_path)

    validation = validate_dataframe(df, require_target=True, project_id=pid, project_spec=spec)
    if not validation.is_valid:
        raise ValueError(f"Training data for '{pid}' failed validation gate: {validation.errors}")

    # Track data version snapshot for this training run
    data_version_info = record_data_version(
        project_id=pid,
        df=df,
        trigger_reason=trigger_reason,
        config=cfg,
    )

    feature_cols = list(spec["features"].keys())
    target_col = spec["target"]

    X = df[feature_cols]
    y = df[target_col].to_numpy(dtype=float)

    seed = int(cfg["data"].get("random_seed", 42))
    test_size = float(cfg["model"].get("test_size", 0.2))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Setup MLflow tracking URI
    tracking_uri = cfg["model"]["mlflow_tracking_uri"]
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        db_rel = tracking_uri.replace("sqlite:///", "")
        db_abs = (PROJECT_ROOT / db_rel).as_posix()
        tracking_uri = f"sqlite:///{db_abs}"

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = f"{cfg['model']['experiment_name']}-{pid}"
    mlflow.set_experiment(experiment_name)

    hyperparams = cfg["model"].get("hyperparameters", {})
    candidates = build_candidate_models(hyperparams, seed=seed)

    leaderboard: list[dict[str, Any]] = []
    trained_objects: dict[str, Any] = {}

    for model_name, estimator in candidates.items():
        t0 = time.perf_counter()
        with mlflow.start_run(run_name=f"{pid}_{model_name}_{trigger_reason}") as run:
            estimator.fit(X_train, y_train)
            duration_s = round(time.perf_counter() - t0, 4)

            preds = estimator.predict(X_test)
            metrics = evaluate_predictions(y_test, preds)
            metrics["train_duration_sec"] = duration_s

            mlflow.log_params(
                {
                    "project_id": pid,
                    "model_name": model_name,
                    "trigger_reason": trigger_reason,
                    "data_version": data_version_info["data_version"],
                    "data_sha256": data_version_info["sha256_short"],
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                }
            )
            mlflow.log_metrics(metrics)

            run_id = run.info.run_id
            leaderboard.append(
                {
                    "model_name": model_name,
                    "model_type": "Ensemble" if model_name == ENSEMBLE_MODEL_NAME else "Single (Dynamic)",
                    "description": MODEL_DESCRIPTIONS.get(model_name, ""),
                    "run_id": run_id,
                    **metrics,
                }
            )
            trained_objects[model_name] = estimator

    # Sort leaderboard by lowest RMSE, then highest R2
    leaderboard.sort(key=lambda item: (item["rmse"], -item["r2"]))
    best_entry = leaderboard[0]
    best_name = best_entry["model_name"]

    # Best individual base model for Dynamic Serving
    base_entries = [row for row in leaderboard if row["model_name"] in BASE_MODEL_NAMES]
    dynamic_best_entry = base_entries[0] if base_entries else best_entry
    dynamic_best_name = dynamic_best_entry["model_name"]

    ensemble_entry = next(
        (row for row in leaderboard if row["model_name"] == ENSEMBLE_MODEL_NAME),
        best_entry,
    )

    min_r2_gate = float(cfg["model"].get("min_r2_gate", 0.70))
    passed_gate = bool(best_entry["r2"] >= min_r2_gate)

    # Determine next semantic model version for this project in Centralized Model Registry
    registry_entries = load_central_registry(cfg)
    proj_entries = [e for e in registry_entries if e.get("project_id") == pid]
    prev_version = max((int(e.get("version_number", 0)) for e in proj_entries), default=0)
    new_version = prev_version + 1 if passed_gate or prev_version == 0 else prev_version
    version_tag = f"v{new_version}.0"

    # Build Dual-Mode Bundle (supports both Dynamic Output and Ensemble Output)
    base_models_dict = {
        name: trained_objects[name] for name in BASE_MODEL_NAMES if name in trained_objects
    }
    bundle = PlugAndPlayModelBundle(
        project_id=pid,
        features=feature_cols,
        target=target_col,
        base_models=base_models_dict,
        ensemble_model=trained_objects[ENSEMBLE_MODEL_NAME],
        dynamic_best_model_name=dynamic_best_name,
        champion_model_name=best_name,
        leaderboard=leaderboard,
        version=version_tag,
    )

    # Save versioned artifact in models/registry/<project_id>_v<N>.joblib
    registry_artifacts_dir = PROJECT_ROOT / "models" / "registry"
    registry_artifacts_dir.mkdir(parents=True, exist_ok=True)
    versioned_artifact_rel = f"models/registry/{pid}_v{new_version}.joblib"
    versioned_artifact_path = PROJECT_ROOT / versioned_artifact_rel
    joblib.dump(bundle, versioned_artifact_path)

    metadata = {
        "registry_id": f"{pid}:{version_tag}",
        "project_id": pid,
        "project_name": spec["name"],
        "domain": spec.get("domain", "ML Project"),
        "model_name": best_name,
        "dynamic_best_model": dynamic_best_name,
        "ensemble_model": ENSEMBLE_MODEL_NAME,
        "registered_model_name": f"{pid}_{cfg['model']['registered_model_name']}",
        "version": version_tag,
        "version_number": new_version,
        "stage": "Production" if passed_gate else "Staging",
        "mlflow_run_id": best_entry["run_id"],
        "experiment_name": experiment_name,
        "promoted": passed_gate,
        "trigger_reason": trigger_reason,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": feature_cols,
        "target": target_col,
        "target_unit": spec.get("target_unit", "units"),
        "data_version": data_version_info,
        "artifact_path": versioned_artifact_rel,
        "metrics": {
            "rmse": best_entry["rmse"],
            "mae": best_entry["mae"],
            "r2": best_entry["r2"],
            "mape_pct": best_entry["mape_pct"],
        },
        "dynamic_metrics": {
            "model_name": dynamic_best_name,
            "rmse": dynamic_best_entry["rmse"],
            "mae": dynamic_best_entry["mae"],
            "r2": dynamic_best_entry["r2"],
        },
        "ensemble_metrics": {
            "model_name": ENSEMBLE_MODEL_NAME,
            "rmse": ensemble_entry["rmse"],
            "mae": ensemble_entry["mae"],
            "r2": ensemble_entry["r2"],
        },
        "leaderboard": leaderboard,
    }

    if passed_gate:
        # Mark older versions of this project as Archived in Centralized Model Registry
        for existing in registry_entries:
            if existing.get("project_id") == pid and existing.get("stage") == "Production":
                existing["stage"] = "Archived"

        registry_entries.append(metadata)
        save_central_registry(registry_entries, cfg)

        # Save per-project active champion artifacts
        proj_champion_path = pdir / "models" / "champion_bundle.joblib"
        proj_meta_path = pdir / "models" / "champion_metadata.json"
        joblib.dump(bundle, proj_champion_path)
        with open(proj_meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Mirror to global champion_artifact_path & metadata_path for DVC & backward compatibility
        champion_path = PROJECT_ROOT / cfg["model"]["champion_artifact_path"]
        metadata_path = PROJECT_ROOT / cfg["model"]["metadata_path"]
        champion_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, champion_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return metadata


if __name__ == "__main__":
    result = train_and_promote(trigger_reason="initial_baseline_training")
    print(
        f"Promoted Champion [{result['project_id']}]: {result['model_name']} ({result['version']}) "
        f"| Dynamic Best={result['dynamic_best_model']} | Ensemble={result['ensemble_model']} "
        f"| R2={result['metrics']['r2']} | DataVersion={result['data_version']['data_version']}"
    )
