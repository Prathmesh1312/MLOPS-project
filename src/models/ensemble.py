"""Simple, Interpretable ML Models, Voting Ensemble & Dual-Mode Serving Bundle."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

BASE_MODEL_NAMES = [
    "LinearRegression",
    "DecisionTree",
    "RandomForest",
    "GradientBoosting",
]

ENSEMBLE_MODEL_NAME = "VotingEnsemble"

MODEL_DESCRIPTIONS: dict[str, str] = {
    "LinearRegression": "Simple linear equation (y = w·x + b) over standardized features — fast & easy to interpret.",
    "DecisionTree": "Single flowchart-like tree of if-else feature splits (max_depth=6).",
    "RandomForest": "Bagging collection of decision trees averaging independent predictions.",
    "GradientBoosting": "Sequential small trees where each tree corrects the previous tree's residual error.",
    "VotingEnsemble": "Ensemble combining LinearRegression + DecisionTree + RandomForest + GradientBoosting.",
}


class PlugAndPlayModelBundle:
    """
    Serializable model bundle supporting both CD serving modes:
      1. Dynamic Output: dynamically routes to the best single model (or requested strategy/model)
      2. Ensemble Output: combines all base models via VotingEnsemble and returns per-model votes + uncertainty
    Also implements standard `.predict(X)` for scikit-learn compatibility.
    """

    def __init__(
        self,
        project_id: str,
        features: list[str],
        target: str,
        base_models: dict[str, Any],
        ensemble_model: Any,
        dynamic_best_model_name: str,
        champion_model_name: str,
        leaderboard: list[dict[str, Any]],
        version: str = "v1.0",
    ) -> None:
        self.project_id = project_id
        self.features = list(features)
        self.target = target
        self.base_models = base_models
        self.ensemble_model = ensemble_model
        self.dynamic_best_model_name = dynamic_best_model_name
        self.champion_model_name = champion_model_name
        self.leaderboard = leaderboard
        self.version = version

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Default scikit-learn compatible predict using the Ensemble model."""
        return np.asarray(self.ensemble_model.predict(X), dtype=float)

    def predict_dynamic(
        self,
        X: pd.DataFrame,
        strategy: str = "best_accuracy",
        preferred_model: str | None = None,
    ) -> dict[str, Any]:
        """
        Dynamic Output Mode:
        Dynamically selects a single base model at inference time based on strategy or explicit choice.
        """
        metrics_by_model = {
            row["model_name"]: row for row in self.leaderboard
        }

        if preferred_model and preferred_model in self.base_models:
            chosen_name = preferred_model
            reason = f"Explicitly requested model '{preferred_model}'"
        elif strategy == "fastest_latency":
            chosen_name = "LinearRegression" if "LinearRegression" in self.base_models else self.dynamic_best_model_name
            reason = "Selected lowest-latency linear model for ultra-fast dynamic inference"
        elif strategy == "interpretable_tree":
            chosen_name = "DecisionTree" if "DecisionTree" in self.base_models else self.dynamic_best_model_name
            reason = "Selected interpretable DecisionTree rule model"
        else:
            chosen_name = self.dynamic_best_model_name
            m_info = metrics_by_model.get(chosen_name, {})
            reason = (
                f"Dynamically selected top-performing single model "
                f"(R²={m_info.get('r2', '--')}, RMSE={m_info.get('rmse', '--')})"
            )

        estimator = self.base_models[chosen_name]
        pred_val = float(estimator.predict(X)[0])
        m_stats = metrics_by_model.get(chosen_name, {})

        return {
            "prediction": round(pred_val, 3),
            "selected_model": chosen_name,
            "selection_strategy": strategy,
            "selection_reason": reason,
            "model_description": MODEL_DESCRIPTIONS.get(chosen_name, ""),
            "validation_r2": m_stats.get("r2"),
            "validation_rmse": m_stats.get("rmse"),
        }

    def predict_ensemble(self, X: pd.DataFrame) -> dict[str, Any]:
        """
        Ensemble Output Mode:
        Evaluates all individual base models + the weighted VotingEnsemble,
        returning the aggregated consensus prediction, every sub-model's vote, and spread.
        """
        member_preds: dict[str, float] = {}
        for name, est in self.base_models.items():
            member_preds[name] = round(float(est.predict(X)[0]), 3)

        ensemble_pred = round(float(self.ensemble_model.predict(X)[0]), 3)
        vals = list(member_preds.values())
        std_dev = round(float(np.std(vals)), 3) if vals else 0.0
        mean_abs = max(abs(ensemble_pred), 1e-3)
        agreement_pct = round(max(0.0, 100.0 - (std_dev / mean_abs) * 100.0), 2)

        metrics_by_model = {row["model_name"]: row for row in self.leaderboard}
        ens_stats = metrics_by_model.get(ENSEMBLE_MODEL_NAME, {})

        return {
            "prediction": ensemble_pred,
            "ensemble_model": ENSEMBLE_MODEL_NAME,
            "member_predictions": member_preds,
            "model_spread_std": std_dev,
            "confidence_interval_95": [
                round(ensemble_pred - 1.96 * std_dev, 3),
                round(ensemble_pred + 1.96 * std_dev, 3),
            ],
            "agreement_score_pct": agreement_pct,
            "validation_r2": ens_stats.get("r2"),
            "validation_rmse": ens_stats.get("rmse"),
        }


def build_candidate_models(hyperparams: dict[str, Any] | None = None, seed: int = 42) -> dict[str, Any]:
    """Construct easy-to-understand individual base models and the VotingEnsemble."""
    hp = hyperparams or {}

    linear = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("linear", LinearRegression()),
        ]
    )

    dt = DecisionTreeRegressor(
        max_depth=int(hp.get("dt_max_depth", 6)),
        min_samples_leaf=4,
        random_state=seed,
    )

    rf = RandomForestRegressor(
        n_estimators=int(hp.get("rf_n_estimators", 80)),
        max_depth=int(hp.get("rf_max_depth", 8)),
        random_state=seed,
        n_jobs=-1,
    )

    gb = GradientBoostingRegressor(
        n_estimators=int(hp.get("gb_n_estimators", 80)),
        learning_rate=float(hp.get("gb_learning_rate", 0.08)),
        max_depth=int(hp.get("gb_max_depth", 4)),
        random_state=seed,
    )

    voting_ensemble = VotingRegressor(
        estimators=[
            ("linear", linear),
            ("dt", dt),
            ("rf", rf),
            ("gb", gb),
        ],
        weights=[1, 1, 2, 2],
    )

    return {
        "LinearRegression": linear,
        "DecisionTree": dt,
        "RandomForest": rf,
        "GradientBoosting": gb,
        "VotingEnsemble": voting_ensemble,
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard regression evaluation metrics (RMSE, MAE, R2, MAPE)."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(
        np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-3, None))) * 100.0
    )
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "mape_pct": round(mape, 3),
    }
