# Model Development and Evaluation Notes

## 1. Project Overview

The Employee Performance Predictor using Data Analytics is an educational machine learning project that explores whether historical, synthetic employee-related indicators can predict a subsequent performance category.

The dataset is synthetically generated. It does not contain verified real-world employee records, and its results must not be interpreted as evidence of real workplace performance.

## 2. Dataset Summary

- Total records: 2,000
- Dataset columns: 26
- Unique employees: 1,000
- Review cycles: 2 per employee
- Prediction target: `perf_band_next`
- Target categories: High, Medium, Low
- Training records: 1,600
- Test records: 400

The data preparation and exploratory data analysis stages checked missing values, duplicate records, invalid values, target distribution, and relationships between selected variables.

## 3. Target Variable

The target column is `perf_band_next`, which represents the next performance category in the synthetic dataset.

The model treats performance prediction as a three-class classification task:

- High
- Medium
- Low

These categories are synthetic labels, not validated employee assessments.

## 4. Feature Preparation

The model uses numeric and categorical input features, including:

- Department, job level, and education
- Experience and manager tenure
- Project counts and task delays
- On-time delivery rate and code review score
- Story points completed and billable-hours ratio
- Training hours and certifications
- Peer feedback and previous manager score
- Kudos, promotions, and salary percentile band

The following columns are excluded from the model features:

- `employee_id`: identifier, not a predictive input
- `review_cycle`: excluded to avoid using the cycle as a predictive feature
- `sick_days`: excluded from model inputs
- `unplanned_absences`: excluded from model inputs
- `avg_login_hours`: excluded from model inputs
- `perf_band_next`: target variable

Excluding sensitive or potentially problematic workplace indicators does not by itself guarantee that the remaining features are fair or free from proxy bias.

## 5. Data Splitting and Leakage Prevention

The project uses `GroupShuffleSplit` with:

- Test size: 20%
- Random state: 42
- Grouping column: `employee_id`

Grouping by employee ensures that records from the same employee do not appear in both the training and test sets.

This produces 1,600 training rows and 400 test rows. The split is reproducible with the same dataset and configuration.

## 6. Preprocessing Pipeline

The preprocessing pipeline handles numeric and categorical data separately.

### Numeric features

- Missing-value imputation using the median
- Standard scaling

### Categorical features

- Missing-value imputation using the most frequent value
- One-hot encoding
- Unknown categories handled using `handle_unknown="ignore"`

Preprocessing is fitted as part of each model pipeline to help prevent information leakage from the test set.

## 7. Models Evaluated

### Logistic Regression

A linear classification model used as a baseline for comparison.

### Random Forest Classifier

An ensemble model using 200 decision trees, a fixed random state, balanced class weights, and parallel processing.

Both models are evaluated on the same employee-grouped test split.

## 8. Evaluation Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.6275 | 0.6000 |
| Macro F1-score | 0.6364 | 0.6074 |

Logistic Regression achieved higher accuracy and macro F1-score on this synthetic holdout dataset.

Macro F1 calculates the F1-score separately for each class and averages the results, giving each class equal weight.

These results are useful for comparing the two models within this experiment. They do not establish real-world predictive accuracy.

## 9. Evaluation Visualizations

The project generates:

- `logistic_regression_confusion_matrix.png`
- `random_forest_confusion_matrix.png`
- `feature_importance_comparison.png`

These files are stored in `outputs/figures/`.

The confusion matrices show correct and incorrect predictions by category.

## 10. Feature Importance

The project exports feature importance values to `outputs/feature_importance.csv`.

- Logistic Regression: mean absolute model coefficients
- Random Forest: impurity-based feature importance

Categorical variables are one-hot encoded, so individual encoded categories may appear as separate features.

Importance values indicate model relationships, not causation. They should not be interpreted as proof that a particular employee characteristic causes better or worse performance.

## 11. Automated Testing

Modeling tests are stored in `tests/test_modeling.py`.

The tests verify:

- Dataset structure and missing-value checks
- Employee-grouped train/test separation
- Metric ranges and model metric availability
- Saved model artifacts
- Feature importance output
- Confusion matrix and comparison chart files
- Exclusion of specified columns from model features

Run the tests using:

```cmd
python -m pytest tests\test_modeling.py -v
```

## 12. Limitations and Responsible Use

- The dataset is synthetic and may not represent real workplaces.
- The target labels and relationships are generated rather than independently validated.
- Results may change on other datasets or under different data-generation assumptions.
- Feature importance does not establish causation.
- The project has not been validated for fairness, employment compliance, or real-world deployment.
- Predictions must not be used to make hiring, promotion, compensation, disciplinary, or termination decisions.
- A real-world application would require representative, lawfully collected data, careful privacy protections, fairness evaluation, independent validation, and meaningful human oversight.

## 13. Conclusion

This project demonstrates a reproducible machine learning workflow covering synthetic data preparation, exploratory data analysis, preprocessing, grouped model evaluation, feature importance analysis, and automated testing.

It is intended for educational demonstration and portfolio development, not as a production employee evaluation system.
