"""Unit tests for plug-and-play dataset generation, data versioning, and schema validation."""
from __future__ import annotations

from src.data.dataset import (
    PROJECT_TEMPLATES,
    build_custom_project_spec,
    compute_dataset_hash,
    generate_project_dataset,
    prepare_datasets,
)
from src.data.validator import validate_dataframe


def test_builtin_project_templates_generate_valid_datasets() -> None:
    for pid, spec in PROJECT_TEMPLATES.items():
        df = generate_project_dataset(project_id=pid, n_samples=120, seed=42, drift_mode=False)
        assert len(df) == 120
        for col in list(spec["features"].keys()) + [spec["target"]]:
            assert col in df.columns
        report = validate_dataframe(df, require_target=True, project_id=pid, project_spec=spec)
        assert report.is_valid is True
        assert report.errors == []


def test_custom_project_spec_and_data_versioning() -> None:
    spec = build_custom_project_spec(
        project_id="laptop_price_test",
        name="Laptop Price Predictor",
        description="Predict laptop price from RAM, SSD, and CPU speed",
        feature_names=["ram_gb", "ssd_gb", "cpu_ghz"],
        target="price_usd",
        target_unit="USD",
    )
    df = generate_project_dataset(
        project_id="laptop_price_test",
        n_samples=100,
        seed=99,
        project_spec=spec,
    )
    assert len(df) == 100
    assert "price_usd" in df.columns
    h1 = compute_dataset_hash(df)
    h2 = compute_dataset_hash(df)
    assert h1 == h2 and len(h1) == 64


def test_schema_validation_catches_out_of_bounds_and_missing_columns() -> None:
    df = generate_project_dataset(project_id="house_price", n_samples=40, seed=42)
    df.loc[0, "bedrooms"] = 99.0  # Outside max bound (10.0)
    df_missing = df.drop(columns=["area_sqft"])

    report = validate_dataframe(df_missing, require_target=True, project_id="house_price")
    assert report.is_valid is False
    assert "area_sqft" in report.missing_columns
    assert report.out_of_bounds_counts.get("bedrooms") == 1


def test_prepare_datasets_records_data_version() -> None:
    prep = prepare_datasets(project_id="house_price")
    assert prep["reference_path"].exists()
    assert prep["processed_train_path"].exists()
    dv = prep["data_version"]
    assert dv["data_version"].startswith("data-v")
    assert len(dv["sha256_short"]) == 12
