"""MLflow Experiment Tracking, Multi-Model Training, and Champion Registry Promotion."""
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
from src.data.dataset import ALL_MODEL_FEATURES, TARGET_COLUMN, prepare_datasets
from src.data.validator import validate_dataframe
from src.models.ensemble import build_candidate_models, evaluate_predictions


def train_and_promote(
    config: dict[str, Any] | None = None,
    trigger_reason: str = "manual_or_ci_pipeline",
    custom_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Train all candidate models + ensembles, track runs in MLflow, and promote the best
    Champion model to disk and MLflow Registry if quality gates pass.
    """
    cfg = config or load_config()
    prepare_datasets(cfg)

    train_csv_path = PROJECT_ROOT / cfg["data"]["processed_train_path"]
    df = custom_df if custom_df is not None else pd.read_csv(train_csv_path)

    validation = validate_dataframe(df, require_target=True)
    if not validation.is_valid:
        raise ValueError(f"Training data failed validation gate: {validation.errors}")

    X = df[ALL_MODEL_FEATURES]
    y = df[TARGET_COLUMN].to_numpy(dtype=float)

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
    experiment_name = cfg["model"]["experiment_name"]
    mlflow.set_experiment(experiment_name)

    hyperparams = cfg["model"]["hyperparameters"]
    candidates = build_candidate_models(hyperparams, seed=seed)

    leaderboard: list[dict[str, Any]] = []
    trained_objects: dict[str, Any] = {}

    for model_name, estimator in candidates.items():
        t0 = time.perf_counter()
        with mlflow.start_run(run_name=f"{model_name}_{trigger_reason}") as run:
            estimator.fit(X_train, y_train)
            duration_s = round(time.perf_counter() - t0, 3)

            preds = estimator.predict(X_test)
            metrics = evaluate_predictions(y_test, preds)
            metrics["train_duration_sec"] = duration_s

            mlflow.log_params(
                {
                    "model_name": model_name,
                    "trigger_reason": trigger_reason,
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                    **hyperparams,
                }
            )
            mlflow.log_metrics(metrics)

            run_id = run.info.run_id
            leaderboard.append(
                {
                    "model_name": model_name,
                    "run_id": run_id,
                    **metrics,
                }
            )
            trained_objects[model_name] = estimator

    # Sort leaderboard by lowest RMSE, preferring StackingEnsemble/VotingEnsemble on ties
    leaderboard.sort(key=lambda item: (item["rmse"], -item["r2"]))
    best_entry = leaderboard[0]
    best_name = best_entry["model_name"]
    best_estimator = trained_objects[best_name]

    min_r2_gate = float(cfg["model"].get("min_r2_gate", 0.85))
    passed_gate = best_entry["r2"] >= min_r2_gate

    champion_path = PROJECT_ROOT / cfg["model"]["champion_artifact_path"]
    metadata_path = PROJECT_ROOT / cfg["model"]["metadata_path"]
    champion_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine next semantic model version
    prev_version = 0
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                old_meta = json.load(f)
                prev_version = int(old_meta.get("version_number", 0))
        except Exception:
            prev_version = 0

    new_version = prev_version + 1 if passed_gate else prev_version

    metadata = {
        "model_name": best_name,
        "registered_model_name": cfg["model"]["registered_model_name"],
        "version": f"v{new_version}.0",
        "version_number": new_version,
        "mlflow_run_id": best_entry["run_id"],
        "experiment_name": experiment_name,
        "promoted": passed_gate,
        "trigger_reason": trigger_reason,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": ALL_MODEL_FEATURES,
        "metrics": {
            "rmse": best_entry["rmse"],
            "mae": best_entry["mae"],
            "r2": best_entry["r2"],
            "mape_pct": best_entry["mape_pct"],
        },
        "leaderboard": leaderboard,
    }

    if passed_gate:
        joblib.dump(best_estimator, champion_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return metadata


if __name__ == "__main__":
    result = train_and_promote(trigger_reason="initial_baseline_training")
    print(
        f"Promoted Champion: {result['model_name']} ({result['version']}) "
        f"| R2={result['metrics']['r2']} | RMSE={result['metrics']['rmse']}"
    )
