"""Generate a reproducible synthetic HR employee-cycle dataset.

This script creates simulated review-cycle records for an educational
performance-band classification project. All values are synthetic.
Records do not represent real employees, and the label is produced by a
transparent scoring formula plus noise — not by a validated HR process.

Protected attributes (gender, race, religion, age, disability, etc.) are
intentionally omitted and are not used to generate the target.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration (documented in docs/data_dictionary.md)
# ---------------------------------------------------------------------------
RNG_SEED = 42
N_EMPLOYEES = 1000
CYCLES_PER_EMPLOYEE = 2
N_RECORDS = N_EMPLOYEES * CYCLES_PER_EMPLOYEE  # 2,000

REVIEW_CYCLES = ("2022-H1", "2022-H2", "2023-H1", "2023-H2", "2024-H1")
DEPARTMENTS = (
    "Engineering",
    "Data",
    "Product",
    "Sales",
    "Operations",
    "Finance",
    "Customer Success",
    "HR",
)
JOB_LEVELS = ("IC1", "IC2", "IC3", "IC4", "M1", "M2")
EDUCATION_LEVELS = ("High School", "Bachelor's", "Master's", "PhD", "Bootcamp")
SALARY_BANDS = ("P10-P25", "P25-P50", "P50-P75", "P75-P90", "P90-P99")

RANGES = {
    "experience_years": (0.5, 25.0),
    "manager_tenure_years": (0.1, 8.0),
    "projects_count": (1, 12),
    "avg_task_delay_days": (0.0, 20.0),
    "on_time_delivery_rate": (0.40, 1.00),
    "bug_count": (0, 25),
    "code_review_score": (1.0, 5.0),
    "story_points_completed": (10, 120),
    "billable_hours_ratio": (0.40, 0.98),
    "training_hours": (0.0, 80.0),
    "certifications_count": (0, 6),
    "internal_hackathons_participated": (0, 4),
    "sick_days": (0, 15),
    "unplanned_absences": (0, 8),
    "avg_login_hours": (5.0, 11.0),
    "peer_feedback_score": (1.0, 5.0),
    "previous_manager_score": (1.0, 5.0),
    "kudos_count": (0, 20),
    "promotions_in_2y": (0, 2),
}

# Target uses only these work, training, and feedback signals.
# Weights sum to 1.0. Columns such as sick_days, avg_login_hours,
# salary band, education, department, and job level are excluded.
SCORE_WEIGHTS = {
    "on_time_delivery_rate": 0.16,
    "delay_inverted": 0.10,
    "story_points_norm": 0.10,
    "bug_inverted": 0.08,
    "code_review_norm": 0.10,
    "billable_hours_ratio": 0.07,
    "projects_norm": 0.05,
    "training_norm": 0.07,
    "certifications_norm": 0.04,
    "hackathons_norm": 0.03,
    "peer_feedback_norm": 0.10,
    "previous_manager_norm": 0.07,
    "kudos_norm": 0.03,
}

NOISE_STD = 0.09
HIGH_THRESHOLD = 0.63
MEDIUM_THRESHOLD = 0.49

COLUMN_ORDER = [
    "employee_id",
    "review_cycle",
    "department",
    "job_level",
    "education",
    "experience_years",
    "manager_tenure_years",
    "projects_count",
    "avg_task_delay_days",
    "on_time_delivery_rate",
    "bug_count",
    "code_review_score",
    "story_points_completed",
    "billable_hours_ratio",
    "training_hours",
    "certifications_count",
    "internal_hackathons_participated",
    "sick_days",
    "unplanned_absences",
    "avg_login_hours",
    "peer_feedback_score",
    "previous_manager_score",
    "kudos_count",
    "promotions_in_2y",
    "salary_percentile_band",
    "perf_band_next",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "employee_performance_synthetic.csv"


def _clip(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(values, low, high)


def _norm(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return (values - low) / (high - low)


def generate_synthetic_dataset(seed: int = RNG_SEED) -> pd.DataFrame:
    """Return 2,000 simulated employee-cycle rows with a noisy 3-class target."""
    rng = np.random.default_rng(seed)

    n = N_EMPLOYEES
    latent = rng.normal(0.0, 1.0, size=n)

    department = rng.choice(
        DEPARTMENTS,
        size=n,
        p=[0.28, 0.12, 0.10, 0.14, 0.12, 0.08, 0.10, 0.06],
    )
    education = rng.choice(
        EDUCATION_LEVELS,
        size=n,
        p=[0.08, 0.52, 0.24, 0.04, 0.12],
    )
    job_level_idx = rng.choice(
        np.arange(len(JOB_LEVELS)),
        size=n,
        p=[0.18, 0.28, 0.24, 0.14, 0.10, 0.06],
    )
    base_experience = _clip(
        3.5 + 3.2 * latent + rng.normal(0.0, 4.0, size=n),
        *RANGES["experience_years"],
    )
    promotions_in_2y = _clip(
        np.round(rng.binomial(2, p=0.12, size=n) + (latent > 1.2).astype(int)),
        *RANGES["promotions_in_2y"],
    ).astype(int)

    start_cycle_idx = rng.integers(0, len(REVIEW_CYCLES) - 1, size=n)

    rows: dict[str, list] = {col: [] for col in COLUMN_ORDER}

    for emp_i in range(n):
        employee_id = f"EMP{emp_i + 1:04d}"
        ability = latent[emp_i]
        mgr_tenure_base = float(
            _clip(rng.uniform(0.2, 4.5) + 0.15 * ability, *RANGES["manager_tenure_years"])
        )

        for cycle_offset in range(CYCLES_PER_EMPLOYEE):
            cycle_idx = int(start_cycle_idx[emp_i] + cycle_offset)
            cycle_noise = rng.normal(0.0, 0.55)

            experience = float(
                _clip(
                    base_experience[emp_i] + 0.5 * cycle_offset,
                    *RANGES["experience_years"],
                )
            )
            manager_tenure = float(
                _clip(mgr_tenure_base + 0.5 * cycle_offset, *RANGES["manager_tenure_years"])
            )

            level_idx = int(
                np.clip(
                    job_level_idx[emp_i] + (cycle_offset == 1 and promotions_in_2y[emp_i] > 0),
                    0,
                    len(JOB_LEVELS) - 1,
                )
            )

            on_time = float(
                _clip(
                    0.78 + 0.10 * ability + 0.08 * cycle_noise + rng.normal(0.0, 0.07),
                    *RANGES["on_time_delivery_rate"],
                )
            )
            delay = float(
                _clip(
                    4.5 - 1.8 * ability - 1.2 * cycle_noise + rng.normal(0.0, 2.2),
                    *RANGES["avg_task_delay_days"],
                )
            )
            bugs = int(
                np.round(
                    _clip(
                        6.0 - 2.2 * ability - 1.0 * cycle_noise + rng.normal(0.0, 3.5),
                        *RANGES["bug_count"],
                    )
                )
            )
            review = float(
                _clip(
                    3.5 + 0.55 * ability + 0.25 * cycle_noise + rng.normal(0.0, 0.45),
                    *RANGES["code_review_score"],
                )
            )
            story = int(
                np.round(
                    _clip(
                        48 + 14 * ability + 8 * cycle_noise + rng.normal(0.0, 16),
                        *RANGES["story_points_completed"],
                    )
                )
            )
            billable = float(
                _clip(
                    0.72 + 0.06 * ability + rng.normal(0.0, 0.08),
                    *RANGES["billable_hours_ratio"],
                )
            )
            projects = int(
                np.round(
                    _clip(
                        4.5 + 1.1 * ability + rng.normal(0.0, 1.8),
                        *RANGES["projects_count"],
                    )
                )
            )
            training = float(
                _clip(
                    18 + 6 * ability + rng.normal(0.0, 14),
                    *RANGES["training_hours"],
                )
            )
            certs = int(
                np.round(
                    _clip(
                        1.1 + 0.7 * ability + rng.normal(0.0, 1.1),
                        *RANGES["certifications_count"],
                    )
                )
            )
            hackathons = int(
                np.round(
                    _clip(
                        0.8 + 0.5 * ability + rng.normal(0.0, 0.9),
                        *RANGES["internal_hackathons_participated"],
                    )
                )
            )
            peer = float(
                _clip(
                    3.5 + 0.50 * ability + 0.20 * cycle_noise + rng.normal(0.0, 0.50),
                    *RANGES["peer_feedback_score"],
                )
            )
            mgr_score = float(
                _clip(
                    3.4 + 0.45 * ability + 0.22 * cycle_noise + rng.normal(0.0, 0.55),
                    *RANGES["previous_manager_score"],
                )
            )
            kudos = int(
                np.round(
                    _clip(
                        5.0 + 2.4 * ability + rng.normal(0.0, 3.2),
                        *RANGES["kudos_count"],
                    )
                )
            )

            # Generated for EDA / later fairness discussion; excluded from the label.
            sick_days = int(rng.integers(0, 16))
            unplanned = int(
                np.round(_clip(rng.poisson(1.2) + rng.integers(0, 2), *RANGES["unplanned_absences"]))
            )
            login_hours = float(
                _clip(rng.normal(8.1, 0.9), *RANGES["avg_login_hours"])
            )

            salary_idx = int(
                np.clip(
                    round(level_idx * 0.7 + experience / 12 + rng.normal(0.0, 0.8)),
                    0,
                    len(SALARY_BANDS) - 1,
                )
            )

            delay_inv = 1.0 - _norm(np.array([delay]), *RANGES["avg_task_delay_days"])[0]
            bug_inv = 1.0 - _norm(np.array([bugs], dtype=float), *RANGES["bug_count"])[0]
            components = {
                "on_time_delivery_rate": on_time,
                "delay_inverted": delay_inv,
                "story_points_norm": _norm(
                    np.array([story], dtype=float), *RANGES["story_points_completed"]
                )[0],
                "bug_inverted": bug_inv,
                "code_review_norm": _norm(
                    np.array([review]), *RANGES["code_review_score"]
                )[0],
                "billable_hours_ratio": billable,
                "projects_norm": _norm(
                    np.array([projects], dtype=float), *RANGES["projects_count"]
                )[0],
                "training_norm": _norm(np.array([training]), *RANGES["training_hours"])[0],
                "certifications_norm": _norm(
                    np.array([certs], dtype=float), *RANGES["certifications_count"]
                )[0],
                "hackathons_norm": _norm(
                    np.array([hackathons], dtype=float),
                    *RANGES["internal_hackathons_participated"],
                )[0],
                "peer_feedback_norm": _norm(
                    np.array([peer]), *RANGES["peer_feedback_score"]
                )[0],
                "previous_manager_norm": _norm(
                    np.array([mgr_score]), *RANGES["previous_manager_score"]
                )[0],
                "kudos_norm": _norm(
                    np.array([kudos], dtype=float), *RANGES["kudos_count"]
                )[0],
            }
            base_score = sum(SCORE_WEIGHTS[k] * components[k] for k in SCORE_WEIGHTS)
            noisy_score = float(_clip(base_score + rng.normal(0.0, NOISE_STD), 0.0, 1.0))

            if noisy_score >= HIGH_THRESHOLD:
                band = "High"
            elif noisy_score >= MEDIUM_THRESHOLD:
                band = "Medium"
            else:
                band = "Low"

            rows["employee_id"].append(employee_id)
            rows["review_cycle"].append(REVIEW_CYCLES[cycle_idx])
            rows["department"].append(department[emp_i])
            rows["job_level"].append(JOB_LEVELS[level_idx])
            rows["education"].append(education[emp_i])
            rows["experience_years"].append(round(experience, 1))
            rows["manager_tenure_years"].append(round(manager_tenure, 1))
            rows["projects_count"].append(projects)
            rows["avg_task_delay_days"].append(round(delay, 1))
            rows["on_time_delivery_rate"].append(round(on_time, 3))
            rows["bug_count"].append(bugs)
            rows["code_review_score"].append(round(review, 1))
            rows["story_points_completed"].append(story)
            rows["billable_hours_ratio"].append(round(billable, 3))
            rows["training_hours"].append(round(training, 1))
            rows["certifications_count"].append(certs)
            rows["internal_hackathons_participated"].append(hackathons)
            rows["sick_days"].append(sick_days)
            rows["unplanned_absences"].append(unplanned)
            rows["avg_login_hours"].append(round(login_hours, 1))
            rows["peer_feedback_score"].append(round(peer, 1))
            rows["previous_manager_score"].append(round(mgr_score, 1))
            rows["kudos_count"].append(kudos)
            rows["promotions_in_2y"].append(int(promotions_in_2y[emp_i]))
            rows["salary_percentile_band"].append(SALARY_BANDS[salary_idx])
            rows["perf_band_next"].append(band)

    frame = pd.DataFrame(rows)[COLUMN_ORDER]
    if len(frame) != N_RECORDS:
        raise ValueError(f"Expected {N_RECORDS} rows, generated {len(frame)}")
    return frame


def validate_dataset(frame: pd.DataFrame) -> dict:
    """Run basic integrity checks. Returns a report dict; does not mutate data."""
    report: dict = {}
    report["shape"] = tuple(frame.shape)
    report["columns"] = list(frame.columns)
    report["missing_total"] = int(frame.isna().sum().sum())
    report["missing_by_column"] = frame.isna().sum().astype(int).to_dict()
    report["duplicate_rows"] = int(frame.duplicated().sum())
    report["duplicate_employee_cycle"] = int(
        frame.duplicated(subset=["employee_id", "review_cycle"]).sum()
    )
    report["unique_employees"] = int(frame["employee_id"].nunique())
    report["cycles_per_employee"] = (
        frame.groupby("employee_id").size().value_counts().sort_index().to_dict()
    )
    report["class_counts"] = frame["perf_band_next"].value_counts().to_dict()
    report["class_percent"] = (
        (frame["perf_band_next"].value_counts(normalize=True) * 100).round(2).to_dict()
    )

    range_failures: list[str] = []
    numeric_specs = {
        "experience_years": RANGES["experience_years"],
        "manager_tenure_years": RANGES["manager_tenure_years"],
        "projects_count": RANGES["projects_count"],
        "avg_task_delay_days": RANGES["avg_task_delay_days"],
        "on_time_delivery_rate": RANGES["on_time_delivery_rate"],
        "bug_count": RANGES["bug_count"],
        "code_review_score": RANGES["code_review_score"],
        "story_points_completed": RANGES["story_points_completed"],
        "billable_hours_ratio": RANGES["billable_hours_ratio"],
        "training_hours": RANGES["training_hours"],
        "certifications_count": RANGES["certifications_count"],
        "internal_hackathons_participated": RANGES["internal_hackathons_participated"],
        "sick_days": RANGES["sick_days"],
        "unplanned_absences": RANGES["unplanned_absences"],
        "avg_login_hours": RANGES["avg_login_hours"],
        "peer_feedback_score": RANGES["peer_feedback_score"],
        "previous_manager_score": RANGES["previous_manager_score"],
        "kudos_count": RANGES["kudos_count"],
        "promotions_in_2y": RANGES["promotions_in_2y"],
    }
    for col, (low, high) in numeric_specs.items():
        series_min, series_max = float(frame[col].min()), float(frame[col].max())
        if series_min < low - 1e-9 or series_max > high + 1e-9:
            range_failures.append(f"{col}: observed [{series_min}, {series_max}] vs [{low}, {high}]")
    report["range_failures"] = range_failures
    report["observed_ranges"] = {
        col: (float(frame[col].min()), float(frame[col].max())) for col in numeric_specs
    }

    expected_cats = {
        "department": set(DEPARTMENTS),
        "job_level": set(JOB_LEVELS),
        "education": set(EDUCATION_LEVELS),
        "salary_percentile_band": set(SALARY_BANDS),
        "review_cycle": set(REVIEW_CYCLES),
        "perf_band_next": {"High", "Medium", "Low"},
    }
    cat_failures: list[str] = []
    for col, allowed in expected_cats.items():
        extra = set(frame[col].unique()) - allowed
        if extra:
            cat_failures.append(f"{col}: unexpected values {sorted(extra)}")
    report["category_failures"] = cat_failures
    report["column_mismatch"] = list(frame.columns) != COLUMN_ORDER
    report["passed"] = (
        report["missing_total"] == 0
        and report["duplicate_rows"] == 0
        and report["duplicate_employee_cycle"] == 0
        and report["shape"] == (N_RECORDS, len(COLUMN_ORDER))
        and not range_failures
        and not cat_failures
        and not report["column_mismatch"]
    )
    return report


def save_dataset(frame: pd.DataFrame, path: Path = OUTPUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def main() -> None:
    frame = generate_synthetic_dataset(RNG_SEED)
    out = save_dataset(frame)
    report = validate_dataset(frame)

    print(f"Wrote {out}")
    print(f"Shape: {report['shape']}")
    print(f"Unique employees: {report['unique_employees']}")
    print(f"Cycles per employee: {report['cycles_per_employee']}")
    print(f"Missing values: {report['missing_total']}")
    print(f"Duplicate rows: {report['duplicate_rows']}")
    print(f"Duplicate employee+cycle keys: {report['duplicate_employee_cycle']}")
    print("Class counts:", report["class_counts"])
    print("Class percent:", report["class_percent"])
    print("Range failures:", report["range_failures"] or "none")
    print("Category failures:", report["category_failures"] or "none")
    print("Validation passed:", report["passed"])


if __name__ == "__main__":
    main()
