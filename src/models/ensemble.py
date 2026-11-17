"""Multi-Model Ensemble Regressor definitions and evaluation metrics."""
from __future__ import annotations

from typing import Any

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def build_candidate_models(hyperparams: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    """Construct base regressors and the multi-model Stacking/Voting Ensembles."""
    ridge = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(hyperparams.get("ridge_alpha", 1.0)))),
        ]
    )

    rf = RandomForestRegressor(
        n_estimators=int(hyperparams.get("rf_n_estimators", 120)),
        max_depth=int(hyperparams.get("rf_max_depth", 10)),
        random_state=seed,
        n_jobs=-1,
    )

    xgb = XGBRegressor(
        n_estimators=int(hyperparams.get("xgb_n_estimators", 140)),
        learning_rate=float(hyperparams.get("xgb_learning_rate", 0.06)),
        max_depth=int(hyperparams.get("xgb_max_depth", 5)),
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )

    lgbm = LGBMRegressor(
        n_estimators=int(hyperparams.get("lgbm_n_estimators", 140)),
        learning_rate=float(hyperparams.get("lgbm_learning_rate", 0.06)),
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )

    voting_ensemble = VotingRegressor(
        estimators=[
            ("rf", rf),
            ("xgb", xgb),
            ("lgbm", lgbm),
            ("ridge", ridge),
        ],
        weights=[2, 3, 3, 1],
    )

    stacking_ensemble = StackingRegressor(
        estimators=[
            ("rf", rf),
            ("xgb", xgb),
            ("lgbm", lgbm),
            ("ridge", ridge),
        ],
        final_estimator=Ridge(alpha=0.5),
        cv=3,
        n_jobs=-1,
    )

    return {
        "RidgeBaseline": ridge,
        "RandomForest": rf,
        "XGBoost": xgb,
        "LightGBM": lgbm,
        "VotingEnsemble": voting_ensemble,
        "StackingEnsemble": stacking_ensemble,
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression evaluation metrics."""
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
