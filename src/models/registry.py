"""Thread-Safe Multi-Project Cloud Model Registry, Dual-Mode CD Serving & Zero-Downtime Manager."""
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
from src.data.dataset import (
    DEFAULT_PROJECT_ID,
    build_custom_project_spec,
    delete_project_workspace,
    get_data_versions,
    get_project_dir,
    get_project_spec,
    list_all_projects,
    prepare_datasets,
    unmark_project_deleted,
)
from src.models.ensemble import PlugAndPlayModelBundle
from src.models.train import (
    load_central_registry,
    save_central_registry,
    train_and_promote,
)


class CloudModelManager:
    """
    Manages plug-and-play ML projects, the Centralized Model Registry,
    Dual-Mode CD Serving (Dynamic Output + Ensemble Output), and atomic
    zero-downtime model hot-swapping across Blue/Green container slots.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._bundles: dict[str, PlugAndPlayModelBundle] = {}
        self._metadata_by_project: dict[str, dict[str, Any]] = {}
        self.active_project_id: str = DEFAULT_PROJECT_ID
        self._loaded_at_utc: str = ""
        self._reload_count: int = 0
        self.container_id: str = os.getenv("HOSTNAME", socket.gethostname())
        self.service_role: str = os.getenv("SERVICE_ROLE", "control-plane")
        self.deployment_slot: str = os.getenv("DEPLOYMENT_SLOT", "blue")

    def ensure_loaded(self, project_id: str | None = None) -> None:
        """Ensure the requested (or active) project has a trained PlugAndPlayModelBundle in memory."""
        with self._lock:
            target_pid = project_id or self.active_project_id
            if target_pid in self._bundles and target_pid in self._metadata_by_project:
                return
            self.reload_model(project_id=target_pid, auto_train_if_missing=True)

    def reload_model(
        self,
        auto_train_if_missing: bool = True,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Atomically load or hot-swap the active project's model bundle from disk."""
        cfg = load_config()
        target_pid = project_id or self.active_project_id or cfg["project"].get("default_project_id", DEFAULT_PROJECT_ID)
        pdir = get_project_dir(target_pid, cfg)
        proj_bundle_path = pdir / "models" / "champion_bundle.joblib"
        proj_meta_path = pdir / "models" / "champion_metadata.json"

        need_train = not proj_bundle_path.exists() or not proj_meta_path.exists()
        loaded_bundle: Any = None
        if not need_train:
            try:
                loaded_bundle = joblib.load(proj_bundle_path)
                if not isinstance(loaded_bundle, PlugAndPlayModelBundle):
                    need_train = True
            except Exception:
                need_train = True

        if need_train:
            if not auto_train_if_missing:
                raise FileNotFoundError(f"Model bundle not found for project '{target_pid}' at {proj_bundle_path}")
            train_and_promote(cfg, trigger_reason="pipeline_init_bootstrap", project_id=target_pid)
            loaded_bundle = joblib.load(proj_bundle_path)

        with open(proj_meta_path, "r", encoding="utf-8") as f:
            new_meta = json.load(f)

        with self._lock:
            self.active_project_id = target_pid
            self._bundles[target_pid] = loaded_bundle
            self._metadata_by_project[target_pid] = new_meta
            self._loaded_at_utc = datetime.now(timezone.utc).isoformat()
            self._reload_count += 1
            self.deployment_slot = os.getenv(
                "DEPLOYMENT_SLOT", cfg["cloud"].get("default_deployment_slot", "blue")
            )

        return self.get_status(project_id=target_pid)

    def switch_project(self, project_id: str) -> dict[str, Any]:
        """Switch the active project in the control plane and load its model bundle."""
        with self._lock:
            self.active_project_id = project_id
            return self.reload_model(auto_train_if_missing=True, project_id=project_id)

    def init_new_project(
        self,
        project_id: str,
        name: str,
        description: str,
        features: list[str],
        target: str,
        target_unit: str = "units",
        domain: str = "Custom ML Project",
    ) -> dict[str, Any]:
        """
        Plug-and-Play Project Initializer:
          1. Creates project config & feature specifications
          2. Generates & versions initial reference/training datasets (data-v1.0 + SHA-256)
          3. Trains candidate models + VotingEnsemble (v1.0)
          4. Registers v1.0 in the Centralized Model Registry
          5. Activates the project in the CD serving engine
        """
        cfg = load_config()
        clean_id = project_id.strip().lower().replace(" ", "_").replace("-", "_")
        unmark_project_deleted(clean_id, cfg)
        pdir = get_project_dir(clean_id, cfg)
        spec_path = pdir / "project_config.json"

        spec = build_custom_project_spec(
            project_id=clean_id,
            name=name,
            description=description,
            feature_names=features,
            target=target,
            target_unit=target_unit,
            domain=domain,
        )
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)

        prepare_datasets(cfg, project_id=clean_id, force_regenerate=True)
        meta = train_and_promote(
            cfg,
            trigger_reason="new_project_pipeline_init",
            project_id=clean_id,
        )
        status = self.switch_project(clean_id)
        return {
            "message": f"Initialized plug-and-play pipeline for project '{clean_id}'",
            "project": spec,
            "initial_model_version": meta,
            "status": status,
        }

    def delete_project(self, project_id: str) -> dict[str, Any]:
        """
        Remove an ML project from the pipeline:
          1. Ensures at least one project remains in the platform
          2. Deletes project workspace & data versions
          3. Removes project artifacts & entries from Centralized Model Registry
          4. Switches active_project_id to a remaining project if the deleted project was active
        """
        cfg = load_config()
        all_projs = list_all_projects(cfg)
        existing_ids = [p["project_id"] for p in all_projs]

        if project_id not in existing_ids:
            raise ValueError(f"Project '{project_id}' does not exist.")
        if len(existing_ids) <= 1:
            raise ValueError("Cannot remove the last remaining project in the pipeline.")

        # Remove registry entries and saved model files for this project
        entries = load_central_registry(cfg)
        kept_entries: list[dict[str, Any]] = []
        for entry in entries:
            if entry.get("project_id") == project_id:
                art_rel = entry.get("artifact_path", "")
                if art_rel:
                    art_path = PROJECT_ROOT / art_rel
                    if art_path.exists():
                        try:
                            art_path.unlink()
                        except Exception:
                            pass
            else:
                kept_entries.append(entry)
        save_central_registry(kept_entries, cfg)

        # Delete workspace directory and mark deleted
        delete_project_workspace(project_id, cfg)

        with self._lock:
            self._bundles.pop(project_id, None)
            self._metadata_by_project.pop(project_id, None)
            remaining = [pid for pid in existing_ids if pid != project_id]
            if self.active_project_id == project_id:
                self.active_project_id = remaining[0]
                self.reload_model(auto_train_if_missing=True, project_id=self.active_project_id)

        return {
            "message": f"Removed project '{project_id}' and its artifacts from the pipeline.",
            "deleted_project_id": project_id,
            "active_project_id": self.active_project_id,
            "status": self.get_status(self.active_project_id),
        }

    def get_central_registry(self, project_id: str | None = None) -> dict[str, Any]:
        """Return all trained model versions from the Centralized Model Registry."""
        self.ensure_loaded()
        cfg = load_config()
        entries = load_central_registry(cfg)
        if project_id:
            filtered = [e for e in entries if e.get("project_id") == project_id]
        else:
            filtered = entries

        filtered_sorted = list(reversed(filtered))
        active_pid = project_id or self.active_project_id
        data_versions = get_data_versions(active_pid, cfg)

        return {
            "active_project_id": self.active_project_id,
            "filter_project_id": project_id,
            "total_versions": len(filtered_sorted),
            "versions": filtered_sorted,
            "data_versions": data_versions,
        }

    def activate_registry_version(self, project_id: str, version: str) -> dict[str, Any]:
        """
        Promote / Rollback to any specific trained model version stored in the Centralized Model Registry
        with zero downtime.
        """
        cfg = load_config()
        entries = load_central_registry(cfg)
        target_entry: dict[str, Any] | None = None

        for entry in entries:
            if entry.get("project_id") == project_id and entry.get("version") == version:
                target_entry = entry
                break

        if target_entry is None:
            raise ValueError(f"Version '{version}' for project '{project_id}' not found in Centralized Model Registry.")

        artifact_rel = target_entry.get("artifact_path", "")
        artifact_path = PROJECT_ROOT / artifact_rel
        if not artifact_path.exists():
            raise FileNotFoundError(f"Saved artifact for {project_id} {version} not found at {artifact_path}")

        bundle = joblib.load(artifact_path)

        for entry in entries:
            if entry.get("project_id") == project_id:
                entry["stage"] = "Production" if entry.get("version") == version else "Archived"
        save_central_registry(entries, cfg)

        target_entry["stage"] = "Production"
        pdir = get_project_dir(project_id, cfg)
        joblib.dump(bundle, pdir / "models" / "champion_bundle.joblib")
        with open(pdir / "models" / "champion_metadata.json", "w", encoding="utf-8") as f:
            json.dump(target_entry, f, indent=2)

        with self._lock:
            self.active_project_id = project_id
            self._bundles[project_id] = bundle
            self._metadata_by_project[project_id] = target_entry
            self._loaded_at_utc = datetime.now(timezone.utc).isoformat()
            self._reload_count += 1

        return self.get_status(project_id=project_id)

    def predict_single(
        self,
        raw_payload: dict[str, float],
        project_id: str | None = None,
        serving_mode: str = "both",
        dynamic_strategy: str = "best_accuracy",
        preferred_model: str | None = None,
    ) -> dict[str, Any]:
        """
        Run thread-safe inference supporting two CD serving modes:
          - 'dynamic': Dynamic Output (routes to best individual model or requested strategy/model)
          - 'ensemble': Ensemble Output (combines all base models via VotingEnsemble + individual breakdown)
          - 'both': Computes and returns both Dynamic Output and Ensemble Output side-by-side
        """
        target_pid = project_id or self.active_project_id
        self.ensure_loaded(target_pid)

        t0 = time.perf_counter()
        spec = get_project_spec(target_pid)
        feature_specs = spec["features"]
        feature_cols = list(feature_specs.keys())

        clean_input: dict[str, float] = {}
        for col in feature_cols:
            if col in raw_payload and raw_payload[col] is not None:
                clean_input[col] = float(raw_payload[col])
            else:
                clean_input[col] = float(feature_specs[col].get("default", 0.0))

        X_df = pd.DataFrame([[clean_input[c] for c in feature_cols]], columns=feature_cols)

        with self._lock:
            bundle = self._bundles[target_pid]
            meta = dict(self._metadata_by_project[target_pid])
            slot = self.deployment_slot
            cid = self.container_id

        dynamic_res = bundle.predict_dynamic(
            X_df, strategy=dynamic_strategy, preferred_model=preferred_model
        )
        ensemble_res = bundle.predict_ensemble(X_df)

        mode = serving_mode.lower().strip()
        if mode == "dynamic":
            primary_pred = dynamic_res["prediction"]
            active_model_used = dynamic_res["selected_model"]
        else:
            primary_pred = ensemble_res["prediction"]
            active_model_used = ensemble_res["ensemble_model"]

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        data_ver = meta.get("data_version", {}).get("data_version", "data-v1.0")

        return {
            "project_id": target_pid,
            "project_name": spec["name"],
            "target_column": spec["target"],
            "target_unit": spec.get("target_unit", "units"),
            "serving_mode": mode,
            "prediction": primary_pred,
            "dynamic_output": dynamic_res,
            "ensemble_output": ensemble_res,
            "features_used": clean_input,
            "model_name": active_model_used,
            "model_version": meta.get("version", "v1.0"),
            "data_version": data_ver,
            "mlflow_run_id": meta.get("mlflow_run_id", ""),
            "deployment_slot": slot,
            "container_id": cid,
            "inference_latency_ms": latency_ms,
        }

    def get_status(self, project_id: str | None = None) -> dict[str, Any]:
        """Return cloud runtime, active project, and model registry state."""
        target_pid = project_id or self.active_project_id
        self.ensure_loaded(target_pid)
        spec = get_project_spec(target_pid)
        with self._lock:
            return {
                "is_ready": target_pid in self._bundles,
                "container_id": self.container_id,
                "service_role": self.service_role,
                "deployment_slot": self.deployment_slot,
                "loaded_at_utc": self._loaded_at_utc,
                "hot_reload_count": self._reload_count,
                "active_project_id": target_pid,
                "active_project": spec,
                "available_projects": list_all_projects(),
                "active_model": self._metadata_by_project.get(target_pid, {}),
            }


model_manager = CloudModelManager()
