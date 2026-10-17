"""Data-quality checks and cleaning for the synthetic employee-cycle dataset.

Run from the project root:

    python -m src.eda

This module does not train models. Valid rows are not dropped or rewritten
except for documented dtype normalization (for example, integer columns stored
as int). Invalid rows, if any, are reported and written aside rather than
silently deleted.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from src.generate_data import (
    COLUMN_ORDER,
    DEPARTMENTS,
    EDUCATION_LEVELS,
    JOB_LEVELS,
    RANGES,
    REVIEW_CYCLES,
    SALARY_BANDS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "employee_performance_synthetic.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "employee_performance_clean.csv"
INVALID_PATH = PROJECT_ROOT / "data" / "processed" / "employee_performance_invalid.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "data_quality_report.json"

EXPECTED_N_ROWS = 2000
EXPECTED_N_EMPLOYEES = 1000
CYCLES_PER_EMPLOYEE = 2
EMPLOYEE_ID_PATTERN = r"^EMP\d{4}$"
TARGET_CLASSES = ("High", "Medium", "Low")
BAND_ORDER = ["Low", "Medium", "High"]

INTEGER_COLUMNS = [
    "projects_count",
    "bug_count",
    "story_points_completed",
    "certifications_count",
    "internal_hackathons_participated",
    "sick_days",
    "unplanned_absences",
    "kudos_count",
    "promotions_in_2y",
]
FLOAT_COLUMNS = [
    "experience_years",
    "manager_tenure_years",
    "avg_task_delay_days",
    "on_time_delivery_rate",
    "code_review_score",
    "billable_hours_ratio",
    "training_hours",
    "avg_login_hours",
    "peer_feedback_score",
    "previous_manager_score",
]
CATEGORICAL_ALLOWED = {
    "review_cycle": set(REVIEW_CYCLES),
    "department": set(DEPARTMENTS),
    "job_level": set(JOB_LEVELS),
    "education": set(EDUCATION_LEVELS),
    "salary_percentile_band": set(SALARY_BANDS),
    "perf_band_next": set(TARGET_CLASSES),
}
NUMERIC_FEATURE_COLUMNS = INTEGER_COLUMNS + FLOAT_COLUMNS


@dataclass
class QualityReport:
    source_path: str
    n_rows_loaded: int
    n_cols_loaded: int
    columns: list[str]
    missing_total: int
    missing_by_column: dict[str, int]
    duplicate_rows: int
    duplicate_employee_cycle: int
    unique_employees: int
    cycles_per_employee: dict[str, int]
    employees_missing_two_cycles: int
    invalid_category_rows: int
    invalid_range_rows: int
    invalid_id_rows: int
    n_invalid_rows: int
    n_excluded_rows: int
    n_value_corrections: int
    n_rows_clean: int
    class_counts: dict[str, int]
    class_percent: dict[str, float]
    issues: list[str] = field(default_factory=list)
    passed: bool = False
    notes: list[str] = field(default_factory=list)


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw synthetic CSV. employee_id stays a string, not an analytic feature."""
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")
    frame = pd.read_csv(path, dtype={"employee_id": "string"})
    return frame


def _normalize_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to documented types without changing valid numeric values."""
    out = frame.copy()
    out["employee_id"] = out["employee_id"].astype("string")
    for col in CATEGORICAL_ALLOWED:
        out[col] = out[col].astype("string")
    for col in INTEGER_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in FLOAT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    # Integer columns that converted cleanly become nullable ints, then plain int if possible.
    for col in INTEGER_COLUMNS:
        if out[col].notna().all() and (out[col] == out[col].round()).all():
            out[col] = out[col].astype(int)
    return out


def _invalid_mask(frame: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    """Return a boolean mask of invalid rows and human-readable issue notes."""
    issues: list[str] = []
    invalid = pd.Series(False, index=frame.index)

    missing_cols = [c for c in COLUMN_ORDER if c not in frame.columns]
    extra_cols = [c for c in frame.columns if c not in COLUMN_ORDER]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        invalid.loc[:] = True
        return invalid, issues
    if extra_cols:
        issues.append(f"Unexpected extra columns (kept but flagged): {extra_cols}")

    bad_id = ~frame["employee_id"].astype(str).str.match(EMPLOYEE_ID_PATTERN, na=False)
    if int(bad_id.sum()) > 0:
        issues.append(f"{int(bad_id.sum())} rows have employee_id outside EMP0001-style IDs.")
        invalid |= bad_id

    for col, allowed in CATEGORICAL_ALLOWED.items():
        bad = ~frame[col].isin(allowed)
        n_bad = int(bad.sum())
        if n_bad:
            extras = sorted(set(frame.loc[bad, col].astype(str).unique()) - allowed)
            issues.append(f"{n_bad} rows have invalid {col} values (examples: {extras[:8]}).")
            invalid |= bad

    for col, (low, high) in RANGES.items():
        numeric = pd.to_numeric(frame[col], errors="coerce")
        bad = numeric.isna() | (numeric < low - 1e-9) | (numeric > high + 1e-9)
        n_bad = int(bad.sum())
        if n_bad:
            issues.append(
                f"{n_bad} rows have {col} outside [{low}, {high}] "
                f"or non-numeric (min={numeric.min()}, max={numeric.max()})."
            )
            invalid |= bad

    return invalid, issues


def audit_dataset(frame: pd.DataFrame, source_path: Path = RAW_PATH) -> QualityReport:
    """Build a data-quality report. Does not drop rows."""
    missing_by_column = frame.isna().sum().astype(int).to_dict()
    cycle_counts = frame.groupby("employee_id").size() if "employee_id" in frame.columns else pd.Series(dtype=int)
    cycles_hist = {str(k): int(v) for k, v in cycle_counts.value_counts().sort_index().items()}
    employees_missing_two = int((cycle_counts != CYCLES_PER_EMPLOYEE).sum()) if len(cycle_counts) else 0

    dup_rows = int(frame.duplicated().sum())
    if {"employee_id", "review_cycle"}.issubset(frame.columns):
        dup_keys = int(frame.duplicated(subset=["employee_id", "review_cycle"]).sum())
    else:
        dup_keys = 0

    invalid, issues = _invalid_mask(frame)
    n_invalid = int(invalid.sum())

    if dup_rows:
        issues.append(f"{dup_rows} fully duplicate rows.")
    if dup_keys:
        issues.append(f"{dup_keys} duplicate employee_id + review_cycle keys.")
    if employees_missing_two:
        issues.append(
            f"{employees_missing_two} employees do not have exactly {CYCLES_PER_EMPLOYEE} cycles."
        )
    if list(frame.columns) != COLUMN_ORDER and not any("Missing required columns" in i for i in issues):
        issues.append("Column order differs from the documented schema (values still checked).")

    class_counts: dict[str, int] = {}
    class_percent: dict[str, float] = {}
    if "perf_band_next" in frame.columns:
        counts = frame["perf_band_next"].value_counts()
        class_counts = {str(k): int(v) for k, v in counts.items()}
        perc = (frame["perf_band_next"].value_counts(normalize=True) * 100).round(2)
        class_percent = {str(k): float(v) for k, v in perc.items()}

    notes = [
        "employee_id is an identifier only and is not an analytical signal.",
        "Valid rows are preserved; dtype normalization does not change valid values.",
        "This dataset is synthetic. Patterns are not evidence about real employees.",
    ]

    report = QualityReport(
        source_path=str(source_path),
        n_rows_loaded=int(len(frame)),
        n_cols_loaded=int(frame.shape[1]),
        columns=list(frame.columns),
        missing_total=int(frame.isna().sum().sum()),
        missing_by_column=missing_by_column,
        duplicate_rows=dup_rows,
        duplicate_employee_cycle=dup_keys,
        unique_employees=int(frame["employee_id"].nunique()) if "employee_id" in frame.columns else 0,
        cycles_per_employee=cycles_hist,
        employees_missing_two_cycles=employees_missing_two,
        invalid_category_rows=0,
        invalid_range_rows=0,
        invalid_id_rows=0,
        n_invalid_rows=n_invalid,
        n_excluded_rows=0,
        n_value_corrections=0,
        n_rows_clean=int(len(frame)),
        class_counts=class_counts,
        class_percent=class_percent,
        issues=issues,
        passed=False,
        notes=notes,
    )

    # Fill specific invalid counters for the printout.
    if "employee_id" in frame.columns:
        report.invalid_id_rows = int(
            (~frame["employee_id"].astype(str).str.match(EMPLOYEE_ID_PATTERN, na=False)).sum()
        )
    cat_bad = pd.Series(False, index=frame.index)
    for col, allowed in CATEGORICAL_ALLOWED.items():
        if col in frame.columns:
            cat_bad |= ~frame[col].isin(allowed)
    report.invalid_category_rows = int(cat_bad.sum())
    range_bad = pd.Series(False, index=frame.index)
    for col, (low, high) in RANGES.items():
        if col in frame.columns:
            numeric = pd.to_numeric(frame[col], errors="coerce")
            range_bad |= numeric.isna() | (numeric < low - 1e-9) | (numeric > high + 1e-9)
    report.invalid_range_rows = int(range_bad.sum())

    report.passed = (
        report.n_rows_loaded == EXPECTED_N_ROWS
        and report.n_cols_loaded == len(COLUMN_ORDER)
        and report.missing_total == 0
        and report.duplicate_rows == 0
        and report.duplicate_employee_cycle == 0
        and report.unique_employees == EXPECTED_N_EMPLOYEES
        and report.employees_missing_two_cycles == 0
        and report.n_invalid_rows == 0
        and set(report.columns) == set(COLUMN_ORDER)
    )
    return report


def clean_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, QualityReport]:
    """Return (clean_frame, invalid_frame, report).

    Valid records are not dropped. Invalid records are separated and reported.
    Duplicate keys, if present, are treated as invalid for modeling hygiene and
    excluded from the clean file with an explicit issue note.
    """
    working = _normalize_dtypes(frame)
    report = audit_dataset(working)
    invalid_mask, _ = _invalid_mask(working)

    if "employee_id" in working.columns and "review_cycle" in working.columns:
        dup_key_mask = working.duplicated(subset=["employee_id", "review_cycle"], keep=False)
    else:
        dup_key_mask = pd.Series(False, index=working.index)
    full_dup_mask = working.duplicated(keep=False)
    exclude_mask = invalid_mask | dup_key_mask | full_dup_mask

    invalid = working.loc[exclude_mask].copy()
    clean = working.loc[~exclude_mask].copy()
    report.n_excluded_rows = int(exclude_mask.sum())
    report.n_rows_clean = int(len(clean))
    report.n_value_corrections = 0

    if report.n_excluded_rows == 0:
        report.notes.append("No invalid or duplicate records found. All loaded rows were written to the clean file.")
    else:
        report.issues.append(
            f"Excluded {report.n_excluded_rows} rows from the clean file "
            f"(invalid values and/or duplicate keys). See {INVALID_PATH.name}."
        )
        report.passed = False

    if "perf_band_next" in clean.columns:
        counts = clean["perf_band_next"].value_counts()
        report.class_counts = {str(k): int(v) for k, v in counts.items()}
        perc = (clean["perf_band_next"].value_counts(normalize=True) * 100).round(2)
        report.class_percent = {str(k): float(v) for k, v in perc.items()}

    return clean, invalid, report


def save_clean_outputs(
    clean: pd.DataFrame,
    invalid: pd.DataFrame,
    report: QualityReport,
    clean_path: Path = CLEAN_PATH,
    invalid_path: Path = INVALID_PATH,
    report_path: Path = REPORT_PATH,
) -> None:
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [c for c in COLUMN_ORDER if c in clean.columns]
    clean[ordered].to_csv(clean_path, index=False)
    if len(invalid):
        invalid.to_csv(invalid_path, index=False)
    elif invalid_path.exists():
        invalid_path.unlink()
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")


def print_quality_report(report: QualityReport) -> None:
    print("=== Data quality report (synthetic HR cycles) ===")
    print(f"Source: {report.source_path}")
    print(f"Loaded shape: {report.n_rows_loaded} rows x {report.n_cols_loaded} columns")
    print(f"Clean rows written: {report.n_rows_clean}")
    print(f"Excluded rows: {report.n_excluded_rows}")
    print(f"Value corrections: {report.n_value_corrections}")
    print(f"Missing cells: {report.missing_total}")
    print(f"Duplicate rows: {report.duplicate_rows}")
    print(f"Duplicate employee+cycle keys: {report.duplicate_employee_cycle}")
    print(f"Unique employees: {report.unique_employees}")
    print(f"Cycles per employee histogram: {report.cycles_per_employee}")
    print(f"Employees without exactly 2 cycles: {report.employees_missing_two_cycles}")
    print(f"Invalid IDs / categories / ranges (row counts): "
          f"{report.invalid_id_rows} / {report.invalid_category_rows} / {report.invalid_range_rows}")
    print("Target counts:", report.class_counts)
    print("Target percent:", report.class_percent)
    print("Issues:", report.issues or "none")
    print("Passed:", report.passed)
    for note in report.notes:
        print(f"Note: {note}")


def run_eda_cleaning() -> QualityReport:
    raw = load_raw()
    clean, invalid, report = clean_dataset(raw)
    save_clean_outputs(clean, invalid, report)
    print_quality_report(report)
    print(f"Wrote clean dataset: {CLEAN_PATH}")
    if report.n_excluded_rows:
        print(f"Wrote excluded rows: {INVALID_PATH}")
    print(f"Wrote JSON report: {REPORT_PATH}")
    return report


def main() -> None:
    run_eda_cleaning()


if __name__ == "__main__":
    main()
