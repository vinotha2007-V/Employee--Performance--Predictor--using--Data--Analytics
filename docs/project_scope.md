# Project scope — Employee Performance Predictor using Data Analytics

## Objective

Build an **educational** analytics project that:

1. Uses a fully **synthetic** HR-style dataset (2,000 simulated employee review-cycle rows).
2. Cleans data and performs exploratory analysis.
3. Predicts the **next-cycle performance band**: `High`, `Medium`, or `Low`.
4. Compares **Logistic Regression** and **Random Forest** in a leakage-safe scikit-learn pipeline.
5. Reports accuracy, macro-F1, a classification report, and a confusion matrix.
6. Explains feature importance and **responsible-AI limitations**, including fairness-oriented checks where appropriate.
7. Ships a Streamlit dashboard for analytics, individual **what-if** predictions, and coaching-style recommendations.

This repository is a beginner-friendly internship / placement portfolio piece. It is **not** a deployed HR decision system.

## Prediction horizon

Each row is one **review cycle**. Features describe signals available at the end of that cycle. The target `perf_band_next` is the simulated band for the **following** cycle.

Later modeling must:

- Keep `perf_band_next` out of the feature matrix.
- Avoid any column that is a direct transform of the hidden label score (that score is not saved in the CSV).
- Prefer employee-aware or time-aware validation because employees appear in **two consecutive cycles**. A random row split can leak identity-level information.

## Educational simulation limitations

- Every field is generated with NumPy (`seed=42`) and documented ranges. There are **no real employees**.
- The label comes from a **transparent weighted formula plus noise**, not from manager judgment, business outcomes, or a calibrated competency model.
- Class overlap is intentional. Perfect accuracy would indicate leakage or an unrealistic label, not a “good HR model.”
- Metrics such as `bug_count` and `code_review_score` are applied across all departments as a simplified schema. Real organizations would not share one identical metric set.
- `previous_manager_score` is a lagged rating. In real data it can encode bias; here it is a simulated number.
- Results will not generalize to any company, job family, or country.

## Privacy considerations

- Do not add real names, emails, employee IDs, or extracts from a real HRIS.
- Do not scrape or blend Glassdoor, LinkedIn, or internal payroll files into this dataset.
- Do not generate **protected attributes** (gender, race, ethnicity, religion, age, disability, sexual orientation, national origin, marital status, and similar).
- If a future fairness exercise needs group labels, use **clearly synthetic, optional demo groups** stored separately, never as production features, and never as drivers of the target.
- Treat login-time and absence fields as sensitive even in simulation: they can resemble surveillance or health data.

## Responsible use rules (required)

Predictions from this project **must not** automatically determine:

- promotion or demotion
- termination or layoff
- disciplinary action
- compensation or bonus
- hiring or “high-potential” program entry

Dashboard copy and documentation must describe outputs as **coaching discussion aids** (for example: training hours, delivery reliability, feedback quality) for a simulated employee. A human decision-maker — and in this project, **no real employment decision** — remains fully responsible.

If stakeholders ask to “rank people to fire” or “auto-promote the High band,” the correct answer is **refuse**: the model is an educational classifier on synthetic labels, not an employment decision engine.

## In-scope vs out-of-scope (this repository)

**In scope**

- Synthetic data generation and a public data dictionary
- EDA, leakage-safe training, model comparison, tests, Streamlit app, README, screenshots

**Out of scope**

- Connecting to live HR systems
- Using protected attributes as features or as label drivers
- Claiming scientific validity, legal compliance (EEOC / local labor law), or production readiness
- Automated adverse actions against people

## Phase boundary

Phase 2 (this documentation + dataset generation) does **not** include model training, the Streamlit app, Git initialization, or package installation unless the project owner requests them.
