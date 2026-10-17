"""Data-quality tests for the synthetic employee performance dataset."""

from __future__ import annotations

import pandas as pd
import pytest

from src.eda import (
    CATEGORICAL_ALLOWED,
    CLEAN_PATH,
    COLUMN_ORDER,
    CYCLES_PER_EMPLOYEE,
    EMPLOYEE_ID_PATTERN,
    EXPECTED_N_EMPLOYEES,
    EXPECTED_N_ROWS,
    RAW_PATH,
    TARGET_CLASSES,
    clean_dataset,
    load_raw,
)
from src.generate_data import RANGES


@pytest.fixture(scope="module")
def raw_frame() -> pd.DataFrame:
    if not RAW_PATH.exists():
        pytest.skip(f"Raw dataset missing: {RAW_PATH}")
    return load_raw(RAW_PATH)


@pytest.fixture(scope="module")
def cleaned(raw_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    return clean_dataset(raw_frame)


def test_expected_shape_and_required_columns(raw_frame: pd.DataFrame) -> None:
    assert raw_frame.shape == (EXPECTED_N_ROWS, len(COLUMN_ORDER))
    assert list(raw_frame.columns) == COLUMN_ORDER


def test_no_missing_values_in_source(raw_frame: pd.DataFrame) -> None:
    assert int(raw_frame.isna().sum().sum()) == 0


def test_no_duplicate_rows_or_employee_cycle_pairs(raw_frame: pd.DataFrame) -> None:
    assert int(raw_frame.duplicated().sum()) == 0
    assert int(raw_frame.duplicated(subset=["employee_id", "review_cycle"]).sum()) == 0


def test_valid_target_and_categorical_values(raw_frame: pd.DataFrame) -> None:
    for col, allowed in CATEGORICAL_ALLOWED.items():
        extras = set(raw_frame[col].unique()) - allowed
        assert not extras, f"Unexpected {col} values: {extras}"
    assert set(raw_frame["perf_band_next"].unique()) <= set(TARGET_CLASSES)


def test_numeric_values_within_documented_ranges(raw_frame: pd.DataFrame) -> None:
    for col, (low, high) in RANGES.items():
        series = pd.to_numeric(raw_frame[col], errors="coerce")
        assert series.notna().all(), f"{col} has non-numeric values"
        assert float(series.min()) >= low - 1e-9, f"{col} min {series.min()} < {low}"
        assert float(series.max()) <= high + 1e-9, f"{col} max {series.max()} > {high}"


def test_both_cycles_retained_for_every_employee(raw_frame: pd.DataFrame) -> None:
    counts = raw_frame.groupby("employee_id").size()
    assert int(raw_frame["employee_id"].nunique()) == EXPECTED_N_EMPLOYEES
    assert counts.min() == CYCLES_PER_EMPLOYEE
    assert counts.max() == CYCLES_PER_EMPLOYEE
    assert raw_frame["employee_id"].astype(str).str.match(EMPLOYEE_ID_PATTERN).all()


def test_clean_output_preserves_rows_and_target_labels(
    raw_frame: pd.DataFrame, cleaned: tuple
) -> None:
    clean, invalid, report = cleaned
    assert report.n_excluded_rows == 0
    assert len(invalid) == 0
    assert len(clean) == len(raw_frame) == EXPECTED_N_ROWS
    assert list(clean.columns) == COLUMN_ORDER
    pd.testing.assert_series_equal(
        clean["perf_band_next"].astype(str).reset_index(drop=True),
        raw_frame["perf_band_next"].astype(str).reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        clean["employee_id"].astype(str).reset_index(drop=True),
        raw_frame["employee_id"].astype(str).reset_index(drop=True),
        check_names=False,
    )
    assert clean.groupby("employee_id").size().min() == CYCLES_PER_EMPLOYEE
    assert clean.groupby("employee_id").size().max() == CYCLES_PER_EMPLOYEE


def test_written_clean_file_matches_in_memory_clean(cleaned: tuple) -> None:
    clean, _, _ = cleaned
    if not CLEAN_PATH.exists():
        pytest.skip("Clean CSV not written yet; run `python -m src.eda` first.")
    on_disk = pd.read_csv(CLEAN_PATH, dtype={"employee_id": "string"})
    assert len(on_disk) == len(clean)
    assert list(on_disk.columns) == COLUMN_ORDER
    assert on_disk["perf_band_next"].astype(str).tolist() == clean["perf_band_next"].astype(str).tolist()
