
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "employee_performance_clean.csv"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"
IMPORTANCE_PATH = ROOT / "outputs" / "feature_importance.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"
MODELS_DIR = ROOT / "models"

TARGET = "perf_band_next"

st.set_page_config(
    page_title="Employee Performance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics():
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_importance():
    return pd.read_csv(IMPORTANCE_PATH)


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


def show_image(filename, caption):
    path = FIGURES_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Visualization not found: {filename}")


try:
    data = load_data()
    metrics = load_metrics()
    importance = load_importance()
except Exception as exc:
    st.error(f"Unable to load project files: {exc}")
    st.stop()


# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Choose a page",
    [
        "Overview",
        "Exploratory Data Analysis",
        "Model Comparison",
        "Feature Importance",
        "Try a Prediction",
        "About Project",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Educational prototype using synthetic data only.")


# Header
st.title("Employee Performance Predictor")
st.caption(
    "Machine Learning • Data Analytics • Model Evaluation"
)

st.warning(
    "Synthetic educational dataset only. Predictions are not validated "
    "employee assessments and must not be used for employment decisions."
)


# Overview
if page == "Overview":
    st.subheader("Project Overview")
    st.write(
        "Explore synthetic employee-related data, performance categories, "
        "model evaluation metrics, and feature importance."
    )

    logistic = metrics["models"]["logistic_regression"]
    forest = metrics["models"]["random_forest"]

    total_rows = len(data)
    employee_count = (
        data["employee_id"].nunique()
        if "employee_id" in data.columns
        else total_rows
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset Records", f"{total_rows:,}")
    col2.metric("Unique Employees", f"{employee_count:,}")
    col3.metric("Logistic Regression Accuracy", f"{logistic['accuracy']:.1%}")
    col4.metric("Random Forest Accuracy", f"{forest['accuracy']:.1%}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Target Category Distribution")
        counts = data[TARGET].value_counts().reindex(
            ["High", "Medium", "Low"], fill_value=0
        )
        st.bar_chart(counts)
        st.dataframe(
            pd.DataFrame({
                "Category": counts.index,
                "Records": counts.values,
                "Percentage": (counts.values / len(data) * 100).round(2),
            }),
            hide_index=True,
            use_container_width=True,
        )

    with right:
        st.subheader("Dataset Preview")
        st.dataframe(data.head(10), use_container_width=True)

    st.subheader("Dataset Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Columns", len(data.columns))
    col2.metric("Missing Values", int(data.isna().sum().sum()))
    col3.metric("Duplicate Rows", int(data.duplicated().sum()))


# EDA
elif page == "Exploratory Data Analysis":
    st.subheader("Exploratory Data Analysis")

    chart_options = {
        "Target Distribution": "target_distribution.png",
        "Numeric Feature Distributions": "numeric_feature_distributions.png",
        "Feature Correlation Heatmap": "feature_correlation_heatmap.png",
        "Performance by Department": "performance_by_department.png",
        "Performance by Job Level": "performance_by_job_level.png",
        "Performance Signal Comparisons": "performance_signal_comparisons.png",
        "Review Cycle Comparison": "review_cycle_comparison.png",
    }

    selected_chart = st.selectbox(
        "Choose a visualization",
        list(chart_options.keys()),
    )

    show_image(
        chart_options[selected_chart],
        selected_chart,
    )

    st.divider()
    st.subheader("Filter Dataset")

    filter_columns = [
        column for column in ["department", "job_level", TARGET]
        if column in data.columns
    ]

    filtered = data.copy()
    filter_cols = st.columns(len(filter_columns))

    for column, container in zip(filter_columns, filter_cols):
        values = sorted(data[column].dropna().astype(str).unique().tolist())
        selected = container.multiselect(
            f"Filter {column.replace('_', ' ').title()}",
            values,
            default=values,
            key=f"filter_{column}",
        )
        filtered = filtered[
            filtered[column].astype(str).isin(selected)
        ]

    st.write(f"Showing {len(filtered):,} of {len(data):,} records.")
    st.dataframe(filtered, use_container_width=True)


# Model comparison
elif page == "Model Comparison":
    st.subheader("Model Performance Comparison")

    logistic = metrics["models"]["logistic_regression"]
    forest = metrics["models"]["random_forest"]

    comparison = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "Random Forest"],
            "Accuracy": [
                logistic["accuracy"],
                forest["accuracy"],
            ],
            "Macro F1-score": [
                logistic["macro_f1"],
                forest["macro_f1"],
            ],
        }
    )

    st.dataframe(
        comparison.style.format({
            "Accuracy": "{:.2%}",
            "Macro F1-score": "{:.2%}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Accuracy")
    st.bar_chart(comparison.set_index("Model")[["Accuracy"]])

    st.subheader("Macro F1-score")
    st.bar_chart(comparison.set_index("Model")[["Macro F1-score"]])

    st.info(
        "Logistic Regression performed better on this particular synthetic "
        "holdout split. This does not establish superiority on real-world data."
    )

    selected_model = st.selectbox(
        "View confusion matrix",
        ["Logistic Regression", "Random Forest"],
    )

    if selected_model == "Logistic Regression":
        show_image(
            "logistic_regression_confusion_matrix.png",
            "Logistic Regression Confusion Matrix",
        )
    else:
        show_image(
            "random_forest_confusion_matrix.png",
            "Random Forest Confusion Matrix",
        )

    st.subheader("Classification Report")
    report_model = (
        logistic
        if selected_model == "Logistic Regression"
        else forest
    )
    st.json(report_model["classification_report"])


# Feature importance
elif page == "Feature Importance":
    st.subheader("Feature Importance Explorer")

    selected_model = st.selectbox(
        "Choose model",
        ["Logistic Regression", "Random Forest"],
    )

    selected_importance = importance[
        importance["model"] == selected_model
    ].copy()

    top_n = st.slider(
        "Number of features to display",
        min_value=5,
        max_value=min(30, len(selected_importance)),
        value=min(12, len(selected_importance)),
    )

    top_features = selected_importance.nlargest(top_n, "importance")
    top_features = top_features.sort_values("importance")

    st.bar_chart(
        top_features.set_index("feature")[["importance"]]
    )

    st.dataframe(
        selected_importance.sort_values(
            "importance", ascending=False
        ),
        hide_index=True,
        use_container_width=True,
    )

    csv_data = selected_importance.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Feature Importance CSV",
        data=csv_data,
        file_name=f"{selected_model.lower().replace(' ', '_')}_feature_importance.csv",
        mime="text/csv",
    )

    st.caption(
        "Feature importance describes how the fitted model uses encoded "
        "features. It does not prove causation or fairness."
    )


# Prediction
elif page == "Try a Prediction":
    st.subheader("Try a Sample Prediction")
    st.write(
        "Enter synthetic sample attributes to see the model's predicted "
        "category. Do not enter real employee personal data."
    )

    model_choice = st.selectbox(
        "Choose prediction model",
        ["Logistic Regression", "Random Forest"],
    )

    model_file = (
        "logistic_regression_pipeline.joblib"
        if model_choice == "Logistic Regression"
        else "random_forest_pipeline.joblib"
    )

    model_path = MODELS_DIR / model_file

    numeric_features = [
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
        "salary_percentile_band",
    ]

    categorical_features = [
        "department",
        "job_level",
        "education",
    ]

    with st.form("prediction_form"):
        st.caption("Numeric fields use example values from the synthetic dataset.")
        inputs = {}

        left, right = st.columns(2)

        for index, column in enumerate(categorical_features):
            options = sorted(
                data[column].dropna().unique().tolist()
            )
            with (left if index % 2 == 0 else right):
                inputs[column] = st.selectbox(
                    column.replace("_", " ").title(),
                    options,
                )

        for index, column in enumerate(numeric_features):
            numeric_series = pd.to_numeric(data[column], errors="coerce")
            minimum = float(numeric_series.min())
            maximum = float(numeric_series.max())
            median = float(numeric_series.median())

            with (left if index % 2 == 0 else right):
                if minimum == maximum:
                    inputs[column] = minimum
                    st.number_input(
                        column.replace("_", " ").title(),
                        value=minimum,
                        disabled=True,
                        key=f"fixed_{column}",
                    )
                else:
                    inputs[column] = st.number_input(
                        column.replace("_", " ").title(),
                        min_value=minimum,
                        max_value=maximum,
                        value=median,
                        key=f"input_{column}",
                    )

        submitted = st.form_submit_button(
            "Predict Synthetic Performance Category",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            model = load_model(str(model_path))
            sample = pd.DataFrame([inputs])
            prediction = model.predict(sample)[0]

            st.subheader("Prediction Result")
            st.metric("Predicted Category", str(prediction))
            st.warning(
                "This is an experimental prediction on synthetic data. "
                "It is not an employee rating or a recommendation for any "
                "workplace decision."
            )

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(sample)[0]
                probability_data = pd.DataFrame({
                    "Category": model.classes_,
                    "Probability": probabilities,
                }).sort_values("Probability", ascending=False)

                st.subheader("Model Class Probabilities")
                st.bar_chart(
                    probability_data.set_index("Category")
                )
                st.dataframe(
                    probability_data.style.format({
                        "Probability": "{:.2%}"
                    }),
                    hide_index=True,
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# About
elif page == "About Project":
    st.subheader("About This Project")

    st.markdown(
        """
        **Project:** Employee Performance Predictor using Data Analytics

        **Methods:** Exploratory data analysis, data preprocessing,
        Logistic Regression, Random Forest, model evaluation, and
        feature importance.

        **Dataset:** 2,000 rows of synthetic educational data.

        **Validation:** Employee-grouped train/test split to keep the same
        employee out of both sets.

        **Purpose:** Demonstrate a reproducible data science workflow.

        **Limitations:** Synthetic data does not validate real workplace
        outcomes. The models are not suitable for employment decisions.
        """
    )

    st.subheader("Project Artifacts")

    artifact_paths = [
        DATA_PATH,
        METRICS_PATH,
        IMPORTANCE_PATH,
        MODELS_DIR / "logistic_regression_pipeline.joblib",
        MODELS_DIR / "random_forest_pipeline.joblib",
    ]

    for artifact in artifact_paths:
        st.write(
            ("✅ " if artifact.exists() else "❌ ")
            + str(artifact.relative_to(ROOT))
        )

    st.subheader("Modeling Documentation")
    notes_path = ROOT / "docs" / "modeling_notes.md"

    if notes_path.exists():
        st.markdown(notes_path.read_text(encoding="utf-8"))
    else:
        st.info("Modeling notes have not been created yet.")
