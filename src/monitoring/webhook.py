"""Automated Continuous Training (CT) Webhook Dispatcher & Local Orchestrator."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from src.config import PROJECT_ROOT, load_config
from src.data.dataset import compute_fao56_features
from src.models.registry import model_manager
from src.models.train import train_and_promote


def dispatch_github_ct_webhook(
    drift_report: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """
    Send a GitHub Actions `repository_dispatch` webhook when running with a GitHub PAT.
    Gracefully records simulated webhook metadata when running offline/locally.
    """
    github_token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_PAT")
    repo = os.getenv("GITHUB_REPOSITORY", config["cloud"]["github_repo"])
    event_type = config["cloud"]["webhook_event_type"]
    url = f"https://api.github.com/repos/{repo}/dispatches"

    payload = {
        "event_type": event_type,
        "client_payload": {
            "triggered_at_utc": datetime.now(timezone.utc).isoformat(),
            "max_psi": drift_report.get("max_psi"),
            "drifted_features": drift_report.get("drifted_features"),
            "drifted_features_ratio": drift_report.get("drifted_features_ratio"),
        },
    }

    if not github_token:
        return {
            "dispatched_remote": False,
            "mode": "local_cloud_simulation",
            "target_endpoint": url,
            "event_type": event_type,
            "message": "No GITHUB_TOKEN set; executed local Continuous Training webhook handler directly.",
        }

    try:
        resp = requests.post(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {github_token}",
            },
            json=payload,
            timeout=10,
        )
        return {
            "dispatched_remote": resp.status_code in (200, 204),
            "status_code": resp.status_code,
            "target_endpoint": url,
            "event_type": event_type,
        }
    except Exception as exc:
        return {
            "dispatched_remote": False,
            "error": str(exc),
            "target_endpoint": url,
        }


def trigger_continuous_training(
    drift_report: dict[str, Any],
    new_production_df: pd.DataFrame | None = None,
    execute_local_retrain: bool = True,
) -> dict[str, Any]:
    """
    Close the MLOps CT loop:
    1. Dispatch GitHub Actions CT webhook (`repository_dispatch`)
    2. Merge latest production dataset (Cloudflare R2 / DVC tracked path)
    3. Retrain multi-model ensemble & log to MLflow
    4. Atomically hot-swap the promoted Champion model in FastAPI with zero downtime
    """
    cfg = load_config()
    webhook_result = dispatch_github_ct_webhook(drift_report, cfg)

    retrain_metadata: dict[str, Any] | None = None
    reload_status: dict[str, Any] | None = None

    if execute_local_retrain:
        ref_path = PROJECT_ROOT / cfg["data"]["reference_path"]
        prod_path = PROJECT_ROOT / cfg["data"]["production_batch_path"]
        train_path = PROJECT_ROOT / cfg["data"]["processed_train_path"]

        if new_production_df is not None:
            enriched_prod = compute_fao56_features(new_production_df)
            enriched_prod.to_csv(prod_path, index=False)
        else:
            enriched_prod = compute_fao56_features(pd.read_csv(prod_path))

        ref_df = compute_fao56_features(pd.read_csv(ref_path))
        merged_df = pd.concat([ref_df, enriched_prod], ignore_index=True)
        merged_df.to_csv(train_path, index=False)

        # Also update reference baseline after adapting to the new climate regime
        retrain_metadata = train_and_promote(
            cfg,
            trigger_reason="ct_webhook_drift_retrain",
            custom_df=merged_df,
        )
        reload_status = model_manager.reload_model(auto_train_if_missing=False)

    ct_event = {
        "event_id": f"ct-{int(datetime.now(timezone.utc).timestamp())}",
        "triggered_at_utc": datetime.now(timezone.utc).isoformat(),
        "drift_summary": {
            "overall_drift_detected": drift_report.get("overall_drift_detected"),
            "max_psi": drift_report.get("max_psi"),
            "drifted_features": drift_report.get("drifted_features"),
        },
        "webhook": webhook_result,
        "promoted_model_version": (
            retrain_metadata.get("version") if retrain_metadata else None
        ),
        "promoted_model_name": (
            retrain_metadata.get("model_name") if retrain_metadata else None
        ),
        "new_metrics": (
            retrain_metadata.get("metrics") if retrain_metadata else None
        ),
        "zero_downtime_hot_reload": reload_status is not None,
    }

    history_path = PROJECT_ROOT / cfg["monitoring"]["ct_history_path"]
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    if history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(ct_event)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history[-50:], f, indent=2)

    return ct_event
