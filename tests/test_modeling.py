
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "employee_performance_clean.csv"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"
IMPORTANCE_PATH = ROOT / "outputs" / "feature_importance.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"
MODELS_DIR = ROOT / "models"

TARGET = "perf_band_next"
GROUP_COLUMN = "employee_id"

EXCLUDED_FEATURES = {
    "employee_id",
    "review_cycle",
    "sick_days",
    "unplanned_absences",
    "avg_login_hours",
    TARGET,
}


def test_clean_dataset_exists_and_has_expected_shape():
    data = pd.read_csv(DATA_PATH)

    assert data.shape == (2000, 26)
    assert TARGET in data.columns
    assert GROUP_COLUMN in data.columns
    assert data.isnull().sum().sum() == 0


def test_train_test_split_has_no_employee_overlap():
    data = pd.read_csv(DATA_PATH)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_indices, test_indices = next(
        splitter.split(data, data[TARGET], groups=data[GROUP_COLUMN])
    )

    train_employees = set(data.iloc[train_indices][GROUP_COLUMN])
    test_employees = set(data.iloc[test_indices][GROUP_COLUMN])

    assert len(train_indices) == 1600
    assert len(test_indices) == 400
    assert train_employees.isdisjoint(test_employees)


def test_metrics_exist_and_are_in_valid_ranges():
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    assert "logistic_regression" in metrics["models"]
    assert "random_forest" in metrics["models"]

    for model_name in ("logistic_regression", "random_forest"):
        model_metrics = metrics["models"][model_name]

        assert 0 <= model_metrics["accuracy"] <= 1
        assert 0 <= model_metrics["macro_f1"] <= 1
        assert len(model_metrics["confusion_matrix"]) == 3


def test_trained_model_artifacts_exist_and_load():
    model_paths = [
        MODELS_DIR / "logistic_regression_pipeline.joblib",
        MODELS_DIR / "random_forest_pipeline.joblib",
    ]

    for model_path in model_paths:
        assert model_path.exists()
        pipeline = joblib.load(model_path)
        assert hasattr(pipeline, "predict")


def test_feature_importance_csv_is_valid():
    importance = pd.read_csv(IMPORTANCE_PATH)

    assert {"model", "feature", "importance", "importance_method"}.issubset(
        importance.columns
    )
    assert not importance.empty
    assert importance["importance"].notna().all()
    assert (importance["importance"] >= 0).all()
    assert set(importance["model"]) == {
        "Logistic Regression",
        "Random Forest",
    }


def test_analysis_figures_exist_and_are_not_empty():
    figure_names = [
        "logistic_regression_confusion_matrix.png",
        "random_forest_confusion_matrix.png",
        "feature_importance_comparison.png",
    ]

    for figure_name in figure_names:
        figure_path = FIGURES_DIR / figure_name
        assert figure_path.exists()
        assert figure_path.stat().st_size > 0


def test_excluded_columns_are_not_model_features():
    data = pd.read_csv(DATA_PATH)
    model_features = set(data.columns) - EXCLUDED_FEATURES

    assert model_features.isdisjoint(EXCLUDED_FEATURES)
    assert TARGET not in model_features
    assert GROUP_COLUMN not in model_features
    assert "review_cycle" not in model_features
    assert "avg_login_hours" not in model_features
