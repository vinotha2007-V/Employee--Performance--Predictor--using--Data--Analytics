
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "employee_performance_clean.csv"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

TARGET = "perf_band_next"
GROUP_COLUMN = "employee_id"
RANDOM_STATE = 42

EXCLUDED_FEATURES = [
    "employee_id",
    "review_cycle",
    "sick_days",
    "unplanned_absences",
    "avg_login_hours",
    TARGET,
]

MODEL_FILES = {
    "Logistic Regression": (
        "logistic_regression",
        MODELS_DIR / "logistic_regression_pipeline.joblib",
        "logistic_regression_confusion_matrix.png",
    ),
    "Random Forest": (
        "random_forest",
        MODELS_DIR / "random_forest_pipeline.joblib",
        "random_forest_confusion_matrix.png",
    ),
}


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_PATH)

    required_columns = {TARGET, GROUP_COLUMN}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns missing from dataset: {sorted(missing_columns)}"
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_indices, test_indices = next(
        splitter.split(data, data[TARGET], groups=data[GROUP_COLUMN])
    )

    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]

    train_employees = set(train_data[GROUP_COLUMN])
    test_employees = set(test_data[GROUP_COLUMN])

    if train_employees & test_employees:
        raise ValueError("Employee leakage detected between train and test sets.")

    feature_columns = [
        column for column in data.columns
        if column not in EXCLUDED_FEATURES
    ]

    X_test = test_data[feature_columns]
    y_test = test_data[TARGET]

    with (OUTPUTS_DIR / "model_metrics.json").open(
        "r", encoding="utf-8"
    ) as file:
        metrics = json.load(file)

    importance_frames = []

    for display_name, (metrics_key, model_path, figure_name) in MODEL_FILES.items():
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        pipeline = joblib.load(model_path)
        predictions = pipeline.predict(X_test)

        # Create confusion matrix
        figure, axis = plt.subplots(figsize=(7, 6))

        display = ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            labels=["High", "Low", "Medium"],
            display_labels=["High", "Low", "Medium"],
            cmap="Blues",
            values_format="d",
            ax=axis,
        )

        display.ax_.set_title(f"{display_name} - Confusion Matrix")
        figure.tight_layout()
        figure.savefig(
            FIGURES_DIR / figure_name,
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(figure)

        # Calculate feature importance
        preprocessor = pipeline.named_steps["preprocessor"]
        transformed_names = preprocessor.get_feature_names_out()
        estimator = pipeline.named_steps["model"]

        if metrics_key == "logistic_regression":
            importance_values = np.mean(np.abs(estimator.coef_), axis=0)
            importance_method = "Mean absolute coefficient"
        else:
            importance_values = estimator.feature_importances_
            importance_method = "Random Forest impurity importance"

        if len(transformed_names) != len(importance_values):
            raise ValueError(
                f"Feature-name mismatch for {display_name}."
            )

        importance_frames.append(
            pd.DataFrame({
                "model": display_name,
                "feature": transformed_names,
                "importance": importance_values,
                "importance_method": importance_method,
            })
        )

        model_metrics = metrics["models"][metrics_key]

        print(f"\n{display_name}")
        print(f"Accuracy: {model_metrics['accuracy']:.4f}")
        print(f"Macro F1: {model_metrics['macro_f1']:.4f}")
        print(f"Confusion matrix saved: {figure_name}")

    # Save feature importance values
    importance_df = pd.concat(importance_frames, ignore_index=True)

    importance_df.to_csv(
        OUTPUTS_DIR / "feature_importance.csv",
        index=False,
    )

    # Compare top 12 features for both models
    figure, axes = plt.subplots(1, 2, figsize=(16, 8))

    for axis, model_name in zip(axes, MODEL_FILES.keys()):
        top_features = (
            importance_df[importance_df["model"] == model_name]
            .nlargest(12, "importance")
            .sort_values("importance")
        )

        axis.barh(
            top_features["feature"],
            top_features["importance"],
        )
        axis.set_title(f"{model_name}: Top 12 Features")
        axis.set_xlabel("Importance")
        axis.set_ylabel("Encoded feature")
        axis.tick_params(axis="y", labelsize=8)

    figure.suptitle(
        "Employee Performance Model Feature Importance\n"
        "Synthetic data; importance does not establish causation"
    )
    figure.tight_layout()

    comparison_path = FIGURES_DIR / "feature_importance_comparison.png"
    figure.savefig(comparison_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print("\nAnalysis completed successfully!")
    print(f"Feature importance CSV: {OUTPUTS_DIR / 'feature_importance.csv'}")
    print(f"Figures directory: {FIGURES_DIR}")
    print(f"Comparison chart: {comparison_path}")


if __name__ == "__main__":
    main()
