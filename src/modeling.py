"""Leakage-safe training of Logistic Regression vs Random Forest.

Run from the project root:

    python -m src.modeling

This is an educational prototype on synthetic labels. It is not a validated
HR decision tool and must not be used to automate promotion, termination,
discipline, or hiring.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.eda import BAND_ORDER, CLEAN_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_PATH = PROJECT_ROOT / "outputs" / "model_metrics.json"
IMPORTANCE_PATH = PROJECT_ROOT / "outputs" / "feature_importance.csv"
MODEL_DIR = PROJECT_ROOT / "models"

TARGET_COL = "perf_band_next"
GROUP_COL = "employee_id"
SPLIT_SEED = 42
MODEL_SEED = 42
TEST_SIZE = 0.20

# Never passed into estimators. GROUP_COL is used only to build disjoint splits.
EXCLUDED_FEATURES = [
    GROUP_COL,
    "review_cycle",
    "sick_days",
    "unplanned_absences",
    "avg_login_hours",
    TARGET_COL,
]

CATEGORICAL_FEATURES = [
    "department",
    "job_level",
    "education",
    "salary_percentile_band",
]
NUMERIC_FEATURES = [
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
    "peer_feedback_score",
    "previous_manager_score",
    "kudos_count",
    "promotions_in_2y",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LOGISTIC_PATH = MODEL_DIR / "logistic_regression_pipeline.joblib"
FOREST_PATH = MODEL_DIR / "random_forest_pipeline.joblib"


def _package_versions() -> dict[str, str]:
    import numpy
    import sklearn

    return {
        "python_sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": numpy.__version__,
        "joblib": joblib.__version__,
    }


def validate_modeling_frame(frame: pd.DataFrame) -> None:
    required = FEATURE_COLUMNS + [TARGET_COL, GROUP_COL]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns before training: {missing}")
    missing_excluded = [c for c in EXCLUDED_FEATURES if c not in frame.columns]
    if missing_excluded:
        raise ValueError(f"Expected columns used for exclusion checks are missing: {missing_excluded}")
    if frame[TARGET_COL].isna().any():
        raise ValueError("Target perf_band_next contains missing values.")
    unknown = set(frame[TARGET_COL].astype(str)) - set(BAND_ORDER)
    if unknown:
        raise ValueError(f"Unexpected target labels: {sorted(unknown)}")
    leaked = [c for c in FEATURE_COLUMNS if c in EXCLUDED_FEATURES]
    if leaked:
        raise ValueError(f"Feature list includes excluded columns: {leaked}")


def load_modeling_frame(path: Path = CLEAN_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Clean dataset not found: {path}. Run python -m src.eda first.")
    frame = pd.read_csv(path, dtype={GROUP_COL: "string"})
    validate_modeling_frame(frame)
    return frame


def make_group_split(
    frame: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = SPLIT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split rows by employee_id so the same person cannot appear in train and test."""
    groups = frame[GROUP_COL]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    dummy_x = np.zeros((len(frame), 1))
    train_idx, test_idx = next(splitter.split(dummy_x, frame[TARGET_COL], groups=groups))
    train = frame.iloc[train_idx].copy()
    test = frame.iloc[test_idx].copy()

    train_ids = set(train[GROUP_COL].astype(str))
    test_ids = set(test[GROUP_COL].astype(str))
    overlap = train_ids & test_ids
    if overlap:
        raise AssertionError(f"Train/test employee IDs overlap: {sorted(overlap)[:10]}")

    summary = {
        "method": "GroupShuffleSplit",
        "group_column": GROUP_COL,
        "test_size": test_size,
        "random_state": random_state,
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_train_employees": int(len(train_ids)),
        "n_test_employees": int(len(test_ids)),
        "disjoint_employees": True,
        "train_target_counts": train[TARGET_COL].value_counts().astype(int).to_dict(),
        "test_target_counts": test[TARGET_COL].value_counts().astype(int).to_dict(),
        "train_target_percent": (train[TARGET_COL].value_counts(normalize=True) * 100).round(2).to_dict(),
        "test_target_percent": (test[TARGET_COL].value_counts(normalize=True) * 100).round(2).to_dict(),
    }
    return train, test, summary


def features_and_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = frame.loc[:, FEATURE_COLUMNS].copy()
    y = frame[TARGET_COL].astype(str)
    banned = [c for c in x.columns if c in EXCLUDED_FEATURES]
    if banned:
        raise ValueError(f"Estimator features contain excluded columns: {banned}")
    if GROUP_COL in x.columns:
        raise ValueError("employee_id leaked into estimator features.")
    return x, y


def build_logistic_pipeline() -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=MODEL_SEED)
    return Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])


def build_forest_pipeline() -> Pipeline:
    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=MODEL_SEED,
        n_jobs=1,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])


def transformed_feature_names(pipeline: Pipeline) -> np.ndarray:
    return pipeline.named_steps["preprocess"].get_feature_names_out()


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    labels = list(BAND_ORDER)
    present = set(pd.Series(y_true).astype(str)) | set(pd.Series(y_pred).astype(str))
    missing_classes = [c for c in labels if c not in present]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "per_class": per_class,
        "classification_report": report,
        "confusion_matrix": {
            "labels": labels,
            "matrix": matrix.astype(int).tolist(),
        },
        "absent_classes_in_y_true_or_pred": missing_classes,
        "n_test_rows": int(len(y_true)),
    }


def plot_confusion_matrix(matrix: list[list[int]], labels: list[str], title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(
        np.array(matrix),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted band")
    ax.set_ylabel("True band")
    ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def extract_feature_importance(pipeline: Pipeline, model_name: str) -> pd.DataFrame:
    names = transformed_feature_names(pipeline)
    readable = []
    is_onehot = []
    for raw in names:
        text = str(raw)
        is_onehot.append(text.startswith("categorical__"))
        readable.append(text.replace("numeric__", "").replace("categorical__", ""))

    clf = pipeline.named_steps["clf"]
    rows: list[dict] = []
    if hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_)
        class_labels = list(clf.classes_)
        mean_abs = np.mean(np.abs(coef), axis=0)
        for feat, onehot, value in zip(readable, is_onehot, mean_abs):
            rows.append(
                {
                    "model": model_name,
                    "feature": feat,
                    "importance": float(value),
                    "method": "mean_abs_logistic_coefficient",
                    "class": "all_classes_mean_abs",
                    "onehot_encoded": bool(onehot),
                }
            )
        for class_i, class_name in enumerate(class_labels):
            for feat, onehot, value in zip(readable, is_onehot, coef[class_i]):
                rows.append(
                    {
                        "model": model_name,
                        "feature": feat,
                        "importance": float(value),
                        "method": "logistic_coefficient",
                        "class": str(class_name),
                        "onehot_encoded": bool(onehot),
                    }
                )
    elif hasattr(clf, "feature_importances_"):
        for feat, onehot, value in zip(readable, is_onehot, clf.feature_importances_):
            rows.append(
                {
                    "model": model_name,
                    "feature": feat,
                    "importance": float(value),
                    "method": "random_forest_gini_importance",
                    "class": "all",
                    "onehot_encoded": bool(onehot),
                }
            )
    else:
        raise TypeError(f"No supported importance method for {type(clf)}")
    return pd.DataFrame(rows)


def plot_feature_importance(importance: pd.DataFrame, path: Path, top_n: int = 12) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    specs = [
        ("logistic_regression", "mean_abs_logistic_coefficient", "Logistic Regression (mean |coef|)"),
        ("random_forest", "random_forest_gini_importance", "Random Forest (Gini importance)"),
    ]
    for ax, (model, method, title) in zip(axes, specs):
        subset = importance[(importance["model"] == model) & (importance["method"] == method)]
        subset = subset.sort_values("importance", ascending=False).head(top_n)
        sns.barplot(data=subset, x="importance", y="feature", color="#4c72b0", ax=ax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Association strength in this synthetic fit")
        ax.set_ylabel("")
    fig.suptitle(
        "Feature importance is association, not causation (synthetic data)",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def train_one(
    name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[Pipeline, dict, list[str], np.ndarray]:
    caught: list[str] = []
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        pipeline.fit(x_train, y_train)
        for item in caught_warnings:
            caught.append(f"{item.category.__name__}: {item.message}")
    y_pred = pipeline.predict(x_test)
    metrics = evaluate_predictions(y_test, y_pred)
    metrics["model"] = name
    metrics["n_features_in"] = int(x_train.shape[1])
    metrics["n_transformed_features"] = int(len(transformed_feature_names(pipeline)))
    metrics["warnings"] = caught
    return pipeline, metrics, caught, y_pred


def run_training(
    *,
    data_path: Path = CLEAN_PATH,
    save_artifacts: bool = True,
    artifact_root: Path | None = None,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")

    root = artifact_root or PROJECT_ROOT
    figure_dir = root / "outputs" / "figures"
    model_dir = root / "models"
    metrics_path = root / "outputs" / "model_metrics.json"
    importance_path = root / "outputs" / "feature_importance.csv"

    frame = load_modeling_frame(data_path)
    train, test, split_summary = make_group_split(frame)
    x_train, y_train = features_and_target(train)
    x_test, y_test = features_and_target(test)

    print("=== Leakage-safe split (GroupShuffleSplit by employee_id) ===")
    print(json.dumps(split_summary, indent=2))
    print("Estimator feature columns:", FEATURE_COLUMNS)
    print("Excluded from estimators:", EXCLUDED_FEATURES)

    fitted = {}
    model_metrics = {}
    importance_frames = []
    confusion_paths = {}

    specs = [
        ("logistic_regression", build_logistic_pipeline(), LOGISTIC_PATH.name),
        ("random_forest", build_forest_pipeline(), FOREST_PATH.name),
    ]
    for name, pipe, filename in specs:
        fitted_pipe, metrics, caught, _y_pred = train_one(
            name, pipe, x_train, y_train, x_test, y_test
        )
        fitted[name] = fitted_pipe
        model_metrics[name] = metrics
        print(f"\n=== {name} ===")
        print(f"accuracy={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}")
        print("per-class:", metrics["per_class"])
        if caught:
            print("warnings:", caught)
        importance_frames.append(extract_feature_importance(fitted_pipe, name))
        if save_artifacts:
            cm_path = figure_dir / f"confusion_matrix_{name}.png"
            plot_confusion_matrix(
                metrics["confusion_matrix"]["matrix"],
                metrics["confusion_matrix"]["labels"],
                f"{name} confusion matrix (synthetic test set)",
                cm_path,
            )
            confusion_paths[name] = str(cm_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(fitted_pipe, model_dir / filename)

    importance = pd.concat(importance_frames, ignore_index=True)
    importance_chart = None
    if save_artifacts:
        figure_dir.mkdir(parents=True, exist_ok=True)
        importance_chart = str(plot_feature_importance(importance, figure_dir / "feature_importance.png"))
        importance_path.parent.mkdir(parents=True, exist_ok=True)
        importance.to_csv(importance_path, index=False)

    payload = {
        "disclaimer": (
            "Educational synthetic-data prototype only. Not a validated HR decision tool. "
            "Do not use predictions to automate promotion, termination, discipline, or hiring. "
            "Feature importance is association within this simulation, not causation."
        ),
        "target": TARGET_COL,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_features": EXCLUDED_FEATURES,
        "group_column_for_split_only": GROUP_COL,
        "split": split_summary,
        "model_random_state": MODEL_SEED,
        "model_config": {
            "logistic_regression": {
                "estimator": "LogisticRegression",
                "max_iter": 2000,
                "solver": "lbfgs",
                "random_state": MODEL_SEED,
                "numeric_preprocess": ["SimpleImputer(median)", "StandardScaler"],
                "categorical_preprocess": [
                    "SimpleImputer(most_frequent)",
                    "OneHotEncoder(handle_unknown=ignore)",
                ],
            },
            "random_forest": {
                "estimator": "RandomForestClassifier",
                "n_estimators": 200,
                "random_state": MODEL_SEED,
                "n_jobs": 1,
                "numeric_preprocess": ["SimpleImputer(median)"],
                "categorical_preprocess": [
                    "SimpleImputer(most_frequent)",
                    "OneHotEncoder(handle_unknown=ignore)",
                ],
            },
        },
        "package_versions": _package_versions(),
        "metrics": model_metrics,
        "artifacts": {
            "metrics_json": str(metrics_path),
            "feature_importance_csv": str(importance_path),
            "logistic_regression_pipeline": str(model_dir / LOGISTIC_PATH.name),
            "random_forest_pipeline": str(model_dir / FOREST_PATH.name),
            "confusion_matrices": confusion_paths,
            "feature_importance_chart": importance_chart,
        },
    }
    if save_artifacts:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {metrics_path}")
        print(f"Wrote {importance_path}")
        print(f"Wrote pipelines under {model_dir}")
    payload["_fitted"] = fitted
    payload["_importance"] = importance
    payload["_train"] = train
    payload["_test"] = test
    payload["_x_train"] = x_train
    payload["_x_test"] = x_test
    payload["_y_train"] = y_train
    payload["_y_test"] = y_test
    return payload


def main() -> None:
    run_training(save_artifacts=True)


if __name__ == "__main__":
    main()
