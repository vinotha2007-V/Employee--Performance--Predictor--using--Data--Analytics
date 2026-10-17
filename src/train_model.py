from pathlib import Path
import json
import warnings
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "employee_performance_clean.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

TARGET = "perf_band_next"
GROUP = "employee_id"
EXCLUDED = {
    "employee_id",
    "review_cycle",
    "sick_days",
    "unplanned_absences",
    "avg_login_hours",
    TARGET,
}
RANDOM_STATE = 42


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}\n"
            "Check that Phase 3 completed successfully."
        )

    df = pd.read_csv(DATA_PATH)

    required = {TARGET, GROUP}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    feature_columns = [
        col for col in df.columns if col not in EXCLUDED
    ]
    if not feature_columns:
        raise ValueError("No model features remain.")

    X = df[feature_columns].copy()
    y = df[TARGET].copy()
    groups = df[GROUP].copy()

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [
        col for col in X.columns if col not in numeric_features
    ]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

    overlap = set(groups_train) & set(groups_test)
    assert not overlap, f"Employee leakage detected: {len(overlap)} overlapping IDs"
    assert GROUP not in X_train.columns
    assert not (EXCLUDED & set(X_train.columns))

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {
        "project_note": (
            "Educational prototype trained on synthetic data only; "
            "not a validated HR decision tool."
        ),
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "rows": len(df),
        "feature_columns": feature_columns,
        "excluded_columns": sorted(EXCLUDED),
        "split": {
            "method": "GroupShuffleSplit",
            "test_size": 0.20,
            "random_state": RANDOM_STATE,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "train_unique_employees": int(groups_train.nunique()),
            "test_unique_employees": int(groups_test.nunique()),
            "overlapping_employee_ids": len(overlap),
            "train_target_counts": y_train.value_counts().to_dict(),
            "test_target_counts": y_test.value_counts().to_dict(),
        },
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "models": {},
    }

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        labels = sorted(y.unique().tolist())

        report = classification_report(
            y_test,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(y_test, predictions, labels=labels)

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(
                f1_score(y_test, predictions, average="macro", zero_division=0)
            ),
            "labels": labels,
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
            "warnings": [str(w.message) for w in caught],
        }

        artifact_path = MODEL_DIR / f"{name}_pipeline.joblib"
        joblib.dump(pipeline, artifact_path)
        metrics["artifact"] = str(artifact_path.relative_to(ROOT))
        all_results["models"][name] = metrics

        print(f"\n{name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro-F1: {metrics['macro_f1']:.4f}")
        print(classification_report(
            y_test, predictions, labels=labels, zero_division=0
        ))
        print("Confusion matrix (label order:", labels, ")")
        print(matrix)

    metrics_path = OUTPUT_DIR / "model_metrics.json"
    metrics_path.write_text(
        json.dumps(all_results, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nSplit summary:")
    print(json.dumps(all_results["split"], indent=2, default=str))
    print(f"\nMetrics saved: {metrics_path}")
    print(f"Models saved in: {MODEL_DIR}")
    print("\nImportant: these metrics describe synthetic data only.")


if __name__ == "__main__":
    main()