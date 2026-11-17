"""Thread-Safe Cloud Model Registry & Zero-Downtime Hot-Swap Manager."""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.data.dataset import ALL_MODEL_FEATURES, compute_single_record_features
from src.models.train import train_and_promote


class CloudModelManager:
    """
    Manages the active ensemble model in memory and supports atomic zero-downtime
    hot-reloading when Continuous Training (CT) promotes a new model artifact.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model: Any = None
        self._metadata: dict[str, Any] = {}
        self._loaded_at_utc: str = ""
        self._reload_count: int = 0
        self.container_id: str = os.getenv("HOSTNAME", socket.gethostname())
        self.deployment_slot: str = os.getenv("DEPLOYMENT_SLOT", "blue")

    def ensure_loaded(self) -> None:
        """Load the champion model from disk or train v1 if not yet present."""
        with self._lock:
            if self._model is not None:
                return
            self.reload_model(auto_train_if_missing=True)

    def reload_model(self, auto_train_if_missing: bool = True) -> dict[str, Any]:
        """Atomically swap the in-memory model with the latest promoted artifact."""
        cfg = load_config()
        champion_path = PROJECT_ROOT / cfg["model"]["champion_artifact_path"]
        metadata_path = PROJECT_ROOT / cfg["model"]["metadata_path"]

        if not champion_path.exists() or not metadata_path.exists():
            if not auto_train_if_missing:
                raise FileNotFoundError(f"Model artifact not found at {champion_path}")
            train_and_promote(cfg, trigger_reason="cold_start_bootstrap")

        new_model = joblib.load(champion_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            new_meta = json.load(f)

        with self._lock:
            self._model = new_model
            self._metadata = new_meta
            self._loaded_at_utc = datetime.now(timezone.utc).isoformat()
            self._reload_count += 1
            self.deployment_slot = os.getenv(
                "DEPLOYMENT_SLOT", cfg["cloud"].get("default_deployment_slot", "blue")
            )

        return self.get_status()

    def predict_single(self, raw_payload: dict[str, float]) -> dict[str, Any]:
        """Run thread-safe inference on a single micro-climate telemetry payload."""
        self.ensure_loaded()
        t0 = time.perf_counter()
        enriched = compute_single_record_features(raw_payload)
        X_df = pd.DataFrame([[enriched[col] for col in ALL_MODEL_FEATURES]], columns=ALL_MODEL_FEATURES)

        with self._lock:
            pred = float(self._model.predict(X_df)[0])
            meta = dict(self._metadata)
            slot = self.deployment_slot
            cid = self.container_id

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return {
            "evaporation_rate_mm_day": round(max(pred, 0.0), 4),
            "fao56_baseline_mm_day": round(enriched["fao56_baseline_mm_day"], 4),
            "vapor_pressure_deficit_kpa": round(enriched["vapor_pressure_deficit_kpa"], 4),
            "model_name": meta.get("model_name", "Unknown"),
            "model_version": meta.get("version", "v1.0"),
            "mlflow_run_id": meta.get("mlflow_run_id", ""),
            "deployment_slot": slot,
            "container_id": cid,
            "inference_latency_ms": latency_ms,
        }

    def get_status(self) -> dict[str, Any]:
        """Return cloud runtime & active model metadata."""
        self.ensure_loaded()
        with self._lock:
            return {
                "is_ready": self._model is not None,
                "container_id": self.container_id,
                "deployment_slot": self.deployment_slot,
                "loaded_at_utc": self._loaded_at_utc,
                "hot_reload_count": self._reload_count,
                "active_model": self._metadata,
            }


model_manager = CloudModelManager()
