# EDA findings — synthetic employee performance data

**Source of numbers:** `data/processed/employee_performance_clean.csv` produced by `python -m src.eda` from `data/raw/employee_performance_synthetic.csv`.  
**Charts:** `outputs/figures/`.  
**Reminder:** every row is simulated. Statistics below describe **this file**, not real employees or a real company.

---

## Dataset overview

| Item | Value |
| --- | --- |
| Grain | One simulated employee review cycle |
| Clean shape | **2,000 rows × 26 columns** |
| Unique `employee_id` | **1,000** |
| Cycles per employee | **2** for every employee |
| Identifier rule | `employee_id` is a key only; it is not a modeling or correlation signal |

Column groups:

- Identifiers: `employee_id`, `review_cycle`
- Role/background (not used to generate the label): department, job level, education, experience, manager tenure, salary band, promotions
- Work / training / feedback signals (used, with noise, to generate the label)
- Health / activity fields **excluded from the label**: `sick_days`, `unplanned_absences`, `avg_login_hours`
- Target: `perf_band_next` ∈ {High, Medium, Low}

---

## Data-quality findings

From `python -m src.eda` on the Phase 2 raw file:

| Check | Result |
| --- | --- |
| Missing cells | **0** |
| Duplicate rows | **0** |
| Duplicate `employee_id` + `review_cycle` | **0** |
| Invalid IDs / categories / ranges | **0 / 0 / 0** |
| Rows excluded | **0** |
| Value corrections | **0** |
| Employees without two cycles | **0** |
| Quality report passed | **True** |

Cleaning **did not delete or rewrite valid values**. The clean CSV is a dtype-normalized copy of all 2,000 source rows, including both cycles per employee. Target labels match the raw file.

---

## Target distribution (actual)

| `perf_band_next` | Count | Percent |
| --- | ---: | ---: |
| Medium | 743 | **37.15%** |
| High | 671 | **33.55%** |
| Low | 586 | **29.30%** |
| Total | 2,000 | 100% |

The classes are mildly imbalanced (Medium is the plurality, Low the smallest). Macro-F1 will still matter in Phase 4 because Low is not rare, but neither is it a tiny class.

---

## Observations (calculated) vs interpretations (not evidence)

Each item separates **what the table shows in this synthetic file** from **a possible story someone might tell**. The story is not proof about real workplaces.

### 1. Delivery signals differ by band, with overlap

**Observation:** Mean `on_time_delivery_rate` is 0.860 (High), 0.774 (Medium), 0.674 (Low). Mean `avg_task_delay_days` is 3.245 (High), 4.456 (Medium), 6.459 (Low). Mean `bug_count` is 4.361 (High) vs 8.155 (Low). Boxplots still overlap across bands (`performance_signal_comparisons.png`).

**Interpretation (possible, not proven):** In a real ops setting, on-time work and fewer defects often sit in the same performance conversation. **Here** the gap exists because the label formula explicitly weights those columns plus noise. It is not independent confirmation that “on-time employees are high performers.”

### 2. Feedback and training move with the band, but not perfectly

**Observation:** Mean `peer_feedback_score` is 3.908 (High) vs 3.008 (Low). Mean `previous_manager_score` is 3.741 (High) vs 2.940 (Low). Mean `training_hours` is 24.252 (High) vs 13.055 (Low). **516 of 1,000** employees (51.6%) have a **different** band on their two cycles.

**Interpretation (possible, not proven):** Feedback and learning effort can co-occur with delivery. Cycle-to-cycle label changes show the generator’s noise: the next-cycle band is not a fixed “person quality” stamp. A model that treated `employee_id` as a feature would cheat on identity, not learn cycle-level signals.

### 3. Sick days and login hours barely move with the target

**Observation:** Mean `sick_days` is 7.568 (High), 7.460 (Medium), 7.573 (Low). Mean `avg_login_hours` is 8.079 (High) vs 8.114 (Low). Pearson correlations of `sick_days` and `avg_login_hours` with other numeric columns are all |r| ≲ 0.06.

**Interpretation (possible, not proven):** Time-in-system and sick leave are weak descriptors of the simulated label, as designed. Using them in a real HR model would still be risky (health and surveillance). **Recommendation for Phase 4:** consider excluding `sick_days`, `unplanned_absences`, and `avg_login_hours` from predictors on responsible-AI grounds, not because they “failed a business KPI.”

### 4. Department mix is uneven; High-band rates differ modestly

**Observation:** Engineering has the most rows (570 / 2,000 = 28.5%). High-band **row share** by department ranges from Operations **38.32%** to Customer Success **29.52%**. The label generator **does not** use department.

**Interpretation (possible, not proven):** In real data this would raise a fairness question (are some teams graded differently?). In **this** file the spread is compatible with random sampling plus the latent ability factor, not a departmental scoring rule. Do not conclude that Operations “outperforms” Customer Success in a real company.

### 5. Numeric work signals are correlated with each other

**Observation:** Strongest Pearson pairs (numeric columns only; `employee_id` excluded) include `on_time_delivery_rate` vs `code_review_score` **r = 0.639**, vs `peer_feedback_score` **r = 0.590**, vs `story_points_completed` **r = 0.589**, and vs `avg_task_delay_days` **r = −0.547**.

**Interpretation (possible, not proven):** Correlated features often share a common cause. Here that cause is the documented latent ability factor plus overlapping formula weights. Correlation is **not** causation: raising story points in the real world would not automatically raise peer scores.

### 6. Job level and education show small rate differences and uneven sample sizes

**Observation:** High-band share is 40.00% at M2 (n = 145 rows) vs 29.65% at IC3 (n = 489). Education is dominated by Bachelor's (1,066 rows). PhD is only **80 rows**, with 47.50% Medium. High-school rows (n = 164) have the highest Low share among education values (35.98%). Education and job level are **not** inputs to the label formula.

**Interpretation (possible, not proven):** Small groups (PhD, M2) produce noisier percentages. A real HR reader might wrongly treat degree or level as performance. For modeling, these fields can become **proxy confounders**. Prefer leakage-safe pipelines and, if used, treat them as descriptive covariates—not as justified “merit” features.

### 7. Review-cycle volume is uneven; 2023-H2 has a higher High share

**Observation:** Cycle counts: 2022-H2 = 537, 2024-H1 = 229, 2022-H1 = 278. High-band share is 38.88% in 2023-H2 vs 30.54% in 2022-H2.

**Interpretation (possible, not proven):** This is not a detected “company improved in 2023.” Employees were assigned a random start cycle. Uneven counts follow from two consecutive cycles inside a five-cycle calendar. Do not time-trend this as organizational performance.

---

## Limitations (required reading)

- Data are **synthetic** (NumPy seed 42). Observed associations largely **recover the generator**, including noise, not a validated competency model.
- No protected attributes are present; that does **not** make a future classifier fair or legal to use on people.
- Do not treat correlations as causal, and do not treat charts as HR evidence.
- Repeated employees mean a **random row split can leak** identity-level information into both train and test.
- `perf_band_next` and any hidden score used to create it must never be features.
- Predictions must not automatically drive promotion, termination, pay, or discipline.

---

## Issues to watch before model training (Phase 4)

1. Use grouped or time-aware validation (employee or later cycle held out), not a naive shuffle of 2,000 rows.
2. Drop `employee_id` from features. Keep it only for grouping.
3. Decide whether department / job level / education / salary band belong in the model at all (they were not label drivers; they can still act as proxies).
4. Prefer excluding `sick_days`, `unplanned_absences`, and `avg_login_hours` from the feature set.
5. Classes are usable as-is; still report **macro-F1** alongside accuracy.
6. Feature correlations among delivery/feedback fields will make coefficient interpretation in logistic regression unstable; Random Forest importances will also cluster related signals.
7. No missing-value imputer is required for this file, but the pipeline should still be leakage-safe for future data.

No blocking data-quality defects were found in the current 2,000 rows.
