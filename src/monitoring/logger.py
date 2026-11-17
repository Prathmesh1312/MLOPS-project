"""Production Data & Cloud Telemetry Logger."""
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
from src.data.dataset import RAW_FEATURES


class CloudTelemetryLogger:
    """
    Logs production inference requests, micro-climate feature distributions,
    container slot metadata, and latency metrics for Cloud Observability & CT Drift checks.
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
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source": source,
            **{k: float(raw_features[k]) for k in RAW_FEATURES if k in raw_features},
            "evaporation_rate_mm_day": prediction_result["evaporation_rate_mm_day"],
            "fao56_baseline_mm_day": prediction_result["fao56_baseline_mm_day"],
            "vapor_pressure_deficit_kpa": prediction_result["vapor_pressure_deficit_kpa"],
            "model_name": prediction_result["model_name"],
            "model_version": prediction_result["model_version"],
            "deployment_slot": prediction_result["deployment_slot"],
            "container_id": prediction_result["container_id"],
            "inference_latency_ms": prediction_result["inference_latency_ms"],
        }

        log_path = self._get_log_path()
        with self._lock:
            self.total_requests += 1
            self._recent_records.append(record)
            self._latencies_ms.append(float(prediction_result["inference_latency_ms"]))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        return record

    def get_recent_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent logged telemetry records."""
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
            items = list(self._recent_records)[-limit:]
            return list(reversed(items))

    def get_logged_features_dataframe(self, max_rows: int = 500) -> pd.DataFrame:
        """Convert recent logged production telemetry into a DataFrame for drift analysis."""
        records = self.get_recent_logs(limit=max_rows)
        if not records:
            return pd.DataFrame(columns=RAW_FEATURES)
        df = pd.DataFrame(records)
        available = [c for c in RAW_FEATURES if c in df.columns]
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
