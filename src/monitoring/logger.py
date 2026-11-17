"""Production Data & Multi-Project Cloud Telemetry Logger."""
from __future__ import annotations

import json
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.data.dataset import DEFAULT_PROJECT_ID, get_project_spec


class CloudTelemetryLogger:
    """
    Logs production inference requests across plug-and-play projects,
    recording Dynamic vs. Ensemble outputs, model & data versions,
    container slot metadata, and latency metrics.
    """

    def __init__(self, max_memory_records: int = 2000) -> None:
        self._lock = threading.RLock()
        self._recent_records: deque[dict[str, Any]] = deque(maxlen=max_memory_records)
        self._latencies_ms: deque[float] = deque(maxlen=max_memory_records)
        self.total_requests: int = 0
        self.total_errors: int = 0
        self.last_drift_psi_max: float = 0.0
        self.last_drift_detected: bool = False

    def _get_log_path(self) -> Path:
        cfg = load_config()
        log_path = PROJECT_ROOT / cfg["monitoring"]["telemetry_log_path"]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return log_path

    def log_prediction(
        self,
        raw_features: dict[str, float],
        prediction_result: dict[str, Any],
        source: str = "api_inference",
    ) -> dict[str, Any]:
        """Record a production inference event to JSONL storage and rolling memory."""
        dyn_out = prediction_result.get("dynamic_output", {})
        ens_out = prediction_result.get("ensemble_output", {})
        primary_pred = prediction_result.get("prediction", 0.0)

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "project_id": prediction_result.get("project_id", DEFAULT_PROJECT_ID),
            "serving_mode": prediction_result.get("serving_mode", "both"),
            "prediction": primary_pred,
            "dynamic_prediction": dyn_out.get("prediction", primary_pred),
            "dynamic_model": dyn_out.get("selected_model", prediction_result.get("model_name", "Unknown")),
            "ensemble_prediction": ens_out.get("prediction", primary_pred),
            "model_name": prediction_result.get("model_name", "VotingEnsemble"),
            "model_version": prediction_result.get("model_version", "v1.0"),
            "data_version": prediction_result.get("data_version", "data-v1.0"),
            "deployment_slot": prediction_result.get("deployment_slot", "blue"),
            "container_id": prediction_result.get("container_id", "local"),
            "inference_latency_ms": float(prediction_result.get("inference_latency_ms", 0.1)),
            "features": {k: float(v) for k, v in raw_features.items() if v is not None},
        }

        log_path = self._get_log_path()
        with self._lock:
            self.total_requests += 1
            self._recent_records.append(record)
            self._latencies_ms.append(record["inference_latency_ms"])
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        return record

    def get_recent_logs(self, limit: int = 50, project_id: str | None = None) -> list[dict[str, Any]]:
        """Return the most recent logged telemetry records (optionally filtered by project_id)."""
        with self._lock:
            if not self._recent_records:
                log_path = self._get_log_path()
                if log_path.exists():
                    with open(log_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    self._recent_records.append(json.loads(line))
                                except Exception:
                                    pass
            items = list(self._recent_records)
            if project_id:
                items = [r for r in items if r.get("project_id") == project_id]
            return list(reversed(items[-limit:]))

    def get_logged_features_dataframe(
        self, project_id: str = DEFAULT_PROJECT_ID, max_rows: int = 500
    ) -> pd.DataFrame:
        """Convert recent logged production telemetry into a DataFrame for drift analysis."""
        spec = get_project_spec(project_id)
        feat_cols = list(spec["features"].keys())
        records = self.get_recent_logs(limit=max_rows, project_id=project_id)
        if not records:
            return pd.DataFrame(columns=feat_cols)
        rows = [r.get("features", {}) for r in records if isinstance(r.get("features"), dict)]
        if not rows:
            return pd.DataFrame(columns=feat_cols)
        df = pd.DataFrame(rows)
        available = [c for c in feat_cols if c in df.columns]
        return df[available]

    def get_observability_summary(self) -> dict[str, Any]:
        """Compute Prometheus-ready cloud observability metrics."""
        with self._lock:
            latencies = list(self._latencies_ms)
            avg_lat = float(np.mean(latencies)) if latencies else 0.0
            p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
            return {
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "buffered_telemetry_records": len(self._recent_records),
                "avg_latency_ms": round(avg_lat, 3),
                "p95_latency_ms": round(p95_lat, 3),
                "last_drift_psi_max": round(self.last_drift_psi_max, 4),
                "last_drift_detected": self.last_drift_detected,
            }


telemetry_logger = CloudTelemetryLogger()
