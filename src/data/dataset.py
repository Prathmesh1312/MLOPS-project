"""Plug-and-Play Multi-Project Dataset Generator, Ingestion & Data Versioning Engine."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config

# ==============================================================================
# Pre-configured Simple & Easy-to-Understand ML Project Templates
# ==============================================================================
PROJECT_TEMPLATES: dict[str, dict[str, Any]] = {
    "house_price": {
        "project_id": "house_price",
        "name": "House Price Prediction",
        "description": "Predict residential property price ($1,000s) from living area, rooms, property age, and distance to city center.",
        "domain": "Real Estate Valuation",
        "target": "price_usd_k",
        "target_unit": "$1,000 USD",
        "features": {
            "area_sqft": {
                "label": "Living Area (sq ft)",
                "min": 300.0,
                "max": 8000.0,
                "default": 1850.0,
                "mean": 1800.0,
                "std": 480.0,
                "drift_shift": 750.0,
                "weight": 0.14,
            },
            "bedrooms": {
                "label": "Bedrooms (count)",
                "min": 1.0,
                "max": 10.0,
                "default": 3.0,
                "mean": 3.2,
                "std": 1.0,
                "drift_shift": 1.4,
                "weight": 18.5,
            },
            "bathrooms": {
                "label": "Bathrooms (count)",
                "min": 1.0,
                "max": 8.0,
                "default": 2.0,
                "mean": 2.2,
                "std": 0.7,
                "drift_shift": 1.0,
                "weight": 22.0,
            },
            "age_years": {
                "label": "Property Age (years)",
                "min": 0.0,
                "max": 100.0,
                "default": 10.0,
                "mean": 18.0,
                "std": 9.0,
                "drift_shift": -8.0,
                "weight": -1.6,
            },
            "distance_city_km": {
                "label": "Distance to City (km)",
                "min": 0.5,
                "max": 80.0,
                "default": 8.5,
                "mean": 12.0,
                "std": 5.0,
                "drift_shift": -4.5,
                "weight": -3.5,
            },
        },
        "base_intercept": 75.0,
        "noise_std": 8.5,
    },
    "student_score": {
        "project_id": "student_score",
        "name": "Student Exam Score Prediction",
        "description": "Predict university final exam marks (0-100) from weekly study hours, class attendance, prior GPA, sleep, and practice tests.",
        "domain": "Education Analytics",
        "target": "exam_score",
        "target_unit": "Marks (0-100)",
        "features": {
            "study_hours_week": {
                "label": "Study Hours / Week",
                "min": 0.0,
                "max": 60.0,
                "default": 16.0,
                "mean": 15.0,
                "std": 5.0,
                "drift_shift": 8.0,
                "weight": 1.55,
            },
            "attendance_pct": {
                "label": "Class Attendance (%)",
                "min": 10.0,
                "max": 100.0,
                "default": 88.0,
                "mean": 82.0,
                "std": 9.0,
                "drift_shift": -15.0,
                "weight": 0.32,
            },
            "previous_gpa": {
                "label": "Previous GPA (0-10)",
                "min": 1.0,
                "max": 10.0,
                "default": 7.8,
                "mean": 7.2,
                "std": 1.1,
                "drift_shift": 1.2,
                "weight": 3.8,
            },
            "sleep_hours": {
                "label": "Daily Sleep (hours)",
                "min": 3.0,
                "max": 12.0,
                "default": 7.2,
                "mean": 7.0,
                "std": 1.0,
                "drift_shift": -1.3,
                "weight": 1.25,
            },
            "practice_tests": {
                "label": "Practice Tests Solved",
                "min": 0.0,
                "max": 25.0,
                "default": 6.0,
                "mean": 5.0,
                "std": 2.0,
                "drift_shift": 3.5,
                "weight": 1.4,
            },
        },
        "base_intercept": 5.0,
        "noise_std": 2.1,
    },
    "cloud_cost": {
        "project_id": "cloud_cost",
        "name": "Cloud Server Cost Estimator",
        "description": "Predict daily cloud infrastructure cost ($ USD) from active users, CPU utilization, RAM, network egress, and storage IOPS.",
        "domain": "Cloud Computing FinOps",
        "target": "daily_cost_usd",
        "target_unit": "USD / Day",
        "features": {
            "active_users": {
                "label": "Concurrent Users",
                "min": 10.0,
                "max": 25000.0,
                "default": 1400.0,
                "mean": 1200.0,
                "std": 350.0,
                "drift_shift": 650.0,
                "weight": 0.03,
            },
            "cpu_utilization_pct": {
                "label": "CPU Utilization (%)",
                "min": 1.0,
                "max": 100.0,
                "default": 64.0,
                "mean": 58.0,
                "std": 14.0,
                "drift_shift": 20.0,
                "weight": 0.95,
            },
            "memory_gb": {
                "label": "Provisioned RAM (GB)",
                "min": 1.0,
                "max": 512.0,
                "default": 32.0,
                "mean": 32.0,
                "std": 10.0,
                "drift_shift": 18.0,
                "weight": 1.5,
            },
            "network_out_gb": {
                "label": "Network Egress (GB)",
                "min": 0.0,
                "max": 5000.0,
                "default": 130.0,
                "mean": 115.0,
                "std": 35.0,
                "drift_shift": 60.0,
                "weight": 0.22,
            },
            "storage_iops": {
                "label": "Storage IOPS",
                "min": 50.0,
                "max": 50000.0,
                "default": 2400.0,
                "mean": 2200.0,
                "std": 600.0,
                "drift_shift": 1100.0,
                "weight": 0.012,
            },
        },
        "base_intercept": 18.0,
        "noise_std": 3.5,
    },
}

DEFAULT_PROJECT_ID = "house_price"
RAW_FEATURES = list(PROJECT_TEMPLATES[DEFAULT_PROJECT_ID]["features"].keys())
ALL_MODEL_FEATURES = list(RAW_FEATURES)
TARGET_COLUMN = PROJECT_TEMPLATES[DEFAULT_PROJECT_ID]["target"]


def get_projects_root(config: dict[str, Any] | None = None) -> Path:
    cfg = config or load_config()
    rel = cfg.get("data", {}).get("projects_dir", "projects")
    root = PROJECT_ROOT / rel
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_project_dir(project_id: str, config: dict[str, Any] | None = None) -> Path:
    pdir = get_projects_root(config) / project_id
    (pdir / "data").mkdir(parents=True, exist_ok=True)
    (pdir / "models").mkdir(parents=True, exist_ok=True)
    return pdir


def build_custom_project_spec(
    project_id: str,
    name: str,
    description: str,
    feature_names: list[str],
    target: str,
    target_unit: str = "units",
    domain: str = "Custom ML Project",
) -> dict[str, Any]:
    """Construct a plug-and-play project specification for any user-defined feature list."""
    features_dict: dict[str, Any] = {}
    for idx, fname in enumerate(feature_names):
        clean_name = fname.strip().lower().replace(" ", "_")
        if not clean_name:
            continue
        sign = -1.0 if ("age" in clean_name or "distance" in clean_name or "loss" in clean_name) else 1.0
        features_dict[clean_name] = {
            "label": clean_name.replace("_", " ").title(),
            "min": 0.0,
            "max": 1000.0,
            "default": round(50.0 + idx * 10.0, 1),
            "mean": round(50.0 + idx * 10.0, 1),
            "std": 15.0,
            "drift_shift": 24.0 * sign,
            "weight": round(sign * (1.8 + 0.4 * idx), 2),
        }

    return {
        "project_id": project_id,
        "name": name,
        "description": description,
        "domain": domain,
        "target": target.strip().lower().replace(" ", "_"),
        "target_unit": target_unit,
        "features": features_dict,
        "base_intercept": 25.0,
        "noise_std": 2.5,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _get_deleted_projects_file(config: dict[str, Any] | None = None) -> Path:
    return get_projects_root(config) / ".deleted_projects.json"


def _load_deleted_project_ids(config: dict[str, Any] | None = None) -> set[str]:
    f = _get_deleted_projects_file(config)
    if not f.exists():
        return set()
    try:
        return set(json.loads(f.read_text(encoding="utf-8")))
    except Exception:
        return set()


def _save_deleted_project_ids(deleted: set[str], config: dict[str, Any] | None = None) -> None:
    f = _get_deleted_projects_file(config)
    f.write_text(json.dumps(sorted(deleted), indent=2), encoding="utf-8")


def unmark_project_deleted(project_id: str, config: dict[str, Any] | None = None) -> None:
    deleted = _load_deleted_project_ids(config)
    if project_id in deleted:
        deleted.discard(project_id)
        _save_deleted_project_ids(deleted, config)


def delete_project_workspace(project_id: str, config: dict[str, Any] | None = None) -> None:
    """Remove a project's workspace directory and mark it deleted."""
    import shutil

    root = get_projects_root(config)
    pdir = root / project_id
    if pdir.exists() and pdir.is_dir():
        shutil.rmtree(pdir)
    deleted = _load_deleted_project_ids(config)
    deleted.add(project_id)
    _save_deleted_project_ids(deleted, config)


def get_project_spec(project_id: str = DEFAULT_PROJECT_ID, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a project's spec from disk or initialize it from built-in templates."""
    pdir = get_project_dir(project_id, config)
    spec_path = pdir / "project_config.json"
    if spec_path.exists():
        with open(spec_path, "r", encoding="utf-8") as f:
            return json.load(f)

    deleted = _load_deleted_project_ids(config)
    if project_id in PROJECT_TEMPLATES and project_id not in deleted:
        spec = dict(PROJECT_TEMPLATES[project_id])
        spec["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)
        return spec

    raise ValueError(f"Unknown project_id '{project_id}'. Initialize it first via /projects/init.")


def list_all_projects(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return metadata for all built-in and user-initialized ML projects."""
    root = get_projects_root(config)
    deleted = _load_deleted_project_ids(config)
    seen: dict[str, dict[str, Any]] = {}

    for pid in PROJECT_TEMPLATES:
        if pid not in deleted:
            spec = get_project_spec(pid, config)
            seen[pid] = spec

    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name not in deleted and (child / "project_config.json").exists():
            try:
                with open(child / "project_config.json", "r", encoding="utf-8") as f:
                    spec = json.load(f)
                    seen[spec["project_id"]] = spec
            except Exception:
                pass

    return list(seen.values())


def generate_project_dataset(
    project_id: str = DEFAULT_PROJECT_ID,
    n_samples: int = 600,
    seed: int = 42,
    drift_mode: bool = False,
    drift_intensity: float = 1.0,
    project_spec: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Generate clean, realistic tabular data for any plug-and-play project specification.
    When drift_mode=True, introduces a distribution shift across features to trigger Continuous Training (CT).
    """
    spec = project_spec or get_project_spec(project_id)
    features_meta: dict[str, dict[str, Any]] = spec["features"]
    target_col: str = spec["target"]
    intercept = float(spec.get("base_intercept", 25.0))
    noise_std = float(spec.get("noise_std", 3.0))

    rng = np.random.default_rng(seed)
    data_cols: dict[str, np.ndarray] = {}
    linear_sum = np.full(n_samples, intercept, dtype=float)

    feature_names = list(features_meta.keys())
    for fname, fcfg in features_meta.items():
        mean = float(fcfg.get("mean", 50.0))
        std = float(fcfg.get("std", 15.0))
        low = float(fcfg.get("min", 0.0))
        high = float(fcfg.get("max", 1000.0))
        shift = float(fcfg.get("drift_shift", std * 1.5)) * drift_intensity if drift_mode else 0.0
        weight = float(fcfg.get("weight", 1.0))

        raw_vals = rng.normal(loc=mean + shift, scale=std, size=n_samples)
        clipped = np.clip(raw_vals, low, high)

        if fname in ("bedrooms", "bathrooms", "practice_tests", "active_users", "storage_iops"):
            clipped = np.round(clipped, 0)
        else:
            clipped = np.round(clipped, 2)

        data_cols[fname] = clipped
        linear_sum += weight * clipped

    df = pd.DataFrame(data_cols)

    # Mild, intuitive non-linear interaction so tree & ensemble models shine alongside Linear Regression
    if len(feature_names) >= 2:
        f0 = df[feature_names[0]].to_numpy(dtype=float)
        f1 = df[feature_names[1]].to_numpy(dtype=float)
        f0_norm = (f0 - np.mean(f0)) / (np.std(f0) + 1e-6)
        f1_norm = (f1 - np.mean(f1)) / (np.std(f1) + 1e-6)
        interaction = (1.8 * noise_std) * (f0_norm * f1_norm + 0.35 * np.sin(f0_norm))
    else:
        interaction = 0.0

    regime_shift = (1.5 * noise_std * drift_intensity) if drift_mode else 0.0
    noise = rng.normal(0.0, noise_std, size=n_samples)
    target_vals = linear_sum + interaction + regime_shift + noise

    if target_col == "exam_score":
        target_vals = np.clip(target_vals, 5.0, 100.0)
    else:
        target_vals = np.maximum(target_vals, 1.0)

    df[target_col] = np.round(target_vals, 2)
    return df


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Compute a deterministic SHA-256 fingerprint for dataset versioning."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def record_data_version(
    project_id: str,
    df: pd.DataFrame,
    trigger_reason: str = "pipeline_init",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record a tracked dataset snapshot (version tag, SHA-256 hash, feature means, DVC link)."""
    pdir = get_project_dir(project_id, config)
    versions_path = pdir / "data" / "data_versions.json"
    history: list[dict[str, Any]] = []
    if versions_path.exists():
        try:
            with open(versions_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    spec = get_project_spec(project_id, config)
    feat_cols = [c for c in spec["features"].keys() if c in df.columns]
    full_hash = compute_dataset_hash(df)
    short_hash = full_hash[:12]

    # Avoid duplicate consecutive entry if data hash is identical
    if history and history[-1].get("sha256_short") == short_hash:
        return history[-1]

    v_num = len(history) + 1
    entry = {
        "data_version": f"data-v{v_num}.0",
        "version_number": v_num,
        "project_id": project_id,
        "sha256_hash": full_hash,
        "sha256_short": short_hash,
        "row_count": len(df),
        "feature_count": len(feat_cols),
        "features": feat_cols,
        "target": spec["target"],
        "feature_means": {
            col: round(float(df[col].mean()), 3) for col in feat_cols
        },
        "trigger_reason": trigger_reason,
        "dvc_tracked": True,
        "storage_tier": "DVC Local Cache + Cloudflare R2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    history.append(entry)
    with open(versions_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return entry


def get_data_versions(project_id: str = DEFAULT_PROJECT_ID, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return all tracked dataset versions for a project (newest first)."""
    pdir = get_project_dir(project_id, config)
    versions_path = pdir / "data" / "data_versions.json"
    if not versions_path.exists():
        return []
    try:
        with open(versions_path, "r", encoding="utf-8") as f:
            items = json.load(f)
            return list(reversed(items))
    except Exception:
        return []


def prepare_datasets(
    config: dict[str, Any] | None = None,
    project_id: str | None = None,
    force_regenerate: bool = False,
) -> dict[str, Any]:
    """
    Ensure reference, production batch, and processed training datasets exist for a project,
    and record the initial data version in the project's lineage store.
    """
    cfg = config or load_config()
    pid = project_id or cfg.get("project", {}).get("default_project_id", DEFAULT_PROJECT_ID)
    spec = get_project_spec(pid, cfg)
    pdir = get_project_dir(pid, cfg)

    proj_ref_path = pdir / "data" / "reference_data.csv"
    proj_prod_path = pdir / "data" / "current_production.csv"
    proj_train_path = pdir / "data" / "train_features.csv"

    n_samples = int(cfg["data"].get("n_reference_samples", 600))
    seed = int(cfg["data"].get("random_seed", 42))

    if force_regenerate or not proj_ref_path.exists():
        ref_df = generate_project_dataset(
            project_id=pid, n_samples=n_samples, seed=seed, drift_mode=False, project_spec=spec
        )
        ref_df.to_csv(proj_ref_path, index=False)
    else:
        ref_df = pd.read_csv(proj_ref_path)

    if force_regenerate or not proj_prod_path.exists():
        prod_df = generate_project_dataset(
            project_id=pid, n_samples=200, seed=seed + 10, drift_mode=False, project_spec=spec
        )
        prod_df.to_csv(proj_prod_path, index=False)
    else:
        prod_df = pd.read_csv(proj_prod_path)

    combined_df = pd.concat([ref_df, prod_df], ignore_index=True)
    combined_df.to_csv(proj_train_path, index=False)

    data_ver = record_data_version(
        project_id=pid,
        df=combined_df,
        trigger_reason="initial_dataset_version",
        config=cfg,
    )

    # Mirror active/default project files to top-level data/ for DVC pipeline compatibility
    global_ref = PROJECT_ROOT / cfg["data"]["reference_path"]
    global_prod = PROJECT_ROOT / cfg["data"]["production_batch_path"]
    global_train = PROJECT_ROOT / cfg["data"]["processed_train_path"]
    global_ref.parent.mkdir(parents=True, exist_ok=True)
    global_train.parent.mkdir(parents=True, exist_ok=True)

    ref_df.to_csv(global_ref, index=False)
    prod_df.to_csv(global_prod, index=False)
    combined_df.to_csv(global_train, index=False)

    return {
        "project_id": pid,
        "reference_path": proj_ref_path,
        "production_batch_path": proj_prod_path,
        "processed_train_path": proj_train_path,
        "data_version": data_ver,
    }


# Backward-compatible helper alias
def generate_microclimate_dataset(
    n_samples: int = 600,
    seed: int = 42,
    drift_mode: bool = False,
    drift_intensity: float = 1.0,
) -> pd.DataFrame:
    """Backward-compatible wrapper generating data for the default project."""
    return generate_project_dataset(
        project_id=DEFAULT_PROJECT_ID,
        n_samples=n_samples,
        seed=seed,
        drift_mode=drift_mode,
        drift_intensity=drift_intensity,
    )


if __name__ == "__main__":
    for template_id in PROJECT_TEMPLATES:
        info = prepare_datasets(project_id=template_id)
        print(
            f"[Init Data] Project={template_id} | Version={info['data_version']['data_version']} "
            f"| Hash={info['data_version']['sha256_short']}"
        )
