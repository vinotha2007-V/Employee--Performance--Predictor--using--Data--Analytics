# Data dictionary — synthetic employee performance dataset

**Status:** All records are **simulated**. They do not describe real employees, real employers, or a validated performance-management process.

**File:** `data/raw/employee_performance_synthetic.csv`  
**Generator:** `src/generate_data.py` (NumPy / Pandas, random seed **42**)  
**Grain:** one row = one simulated employee **review cycle**  
**Size:** 2,000 rows; 1,000 unique `employee_id` values; **two consecutive cycles** per employee so later phases can use time-aware splits.

Protected attributes (gender, race, ethnicity, religion, age, disability, sexual orientation, national origin, and similar) are **not generated** and are **not used** in the label formula.

---

## Identifier and cycle fields

| Column | Type | Meaning | Units | Permitted values / range | Synthetic assumption |
| --- | --- | --- | --- | --- | --- |
| `employee_id` | string | Anonymous simulated employee key | n/a | `EMP0001`–`EMP1000` | Stable across an employee’s cycles. Not a real HRIS ID. |
| `review_cycle` | string | Half-year review window | n/a | `2022-H1`, `2022-H2`, `2023-H1`, `2023-H2`, `2024-H1` | Each employee is assigned a random start cycle and the next consecutive cycle. |

---

## Role and background (not used in the label)

| Column | Type | Meaning | Units | Permitted values / range | Synthetic assumption |
| --- | --- | --- | --- | --- | --- |
| `department` | category | Simulated org unit | n/a | Engineering, Data, Product, Sales, Operations, Finance, Customer Success, HR | Drawn with unequal probabilities (Engineering most common). **Not** an input to `perf_band_next`. |
| `job_level` | category | Career level | n/a | IC1–IC4, M1, M2 | Mostly stable; may increase by one step on the second cycle if `promotions_in_2y` > 0. **Not** used in the label. |
| `education` | category | Highest credential (simulated) | n/a | High School, Bachelor's, Master's, PhD, Bootcamp | Employee-level, constant across cycles. **Not** used in the label. |
| `experience_years` | float | Years of professional experience | years | 0.5–25.0 | Employee baseline plus +0.5 on the later cycle. |
| `manager_tenure_years` | float | Time with current manager | years | 0.1–8.0 | Increases by 0.5 between an employee’s two cycles. |
| `salary_percentile_band` | category | Simulated pay percentile band | n/a | P10-P25, P25-P50, P50-P75, P75-P90, P90-P99 | Loosely related to level and experience. **Not** used in the label. |
| `promotions_in_2y` | integer | Promotions in the prior two years | count | 0–2 | Employee-level. **Not** used in the label. |

---

## Work-delivery signals (used in the label)

Values are clipped to the ranges below. They share a latent “ability” factor plus independent noise so classes overlap.

| Column | Type | Meaning | Units | Permitted range | Synthetic assumption |
| --- | --- | --- | --- | --- | --- |
| `projects_count` | integer | Projects touched in the cycle | count | 1–12 | Higher ability tends to mean slightly more projects. |
| `avg_task_delay_days` | float | Mean days tasks finished after plan | days | 0.0–20.0 | Higher ability tends to mean lower delay. |
| `on_time_delivery_rate` | float | Share of commitments finished on time | proportion 0–1 | 0.40–1.00 | Higher ability tends to mean higher on-time rate. |
| `bug_count` | integer | Defects attributed in the cycle | count | 0–25 | Interpreted as a delivery-quality proxy for all departments in this simulation (simplified schema). |
| `code_review_score` | float | Review / work-quality rating | 1–5 scale | 1.0–5.0 | Named for engineering familiarity; used as a generic quality score for every department. |
| `story_points_completed` | integer | Throughput proxy | points | 10–120 | Generic output volume, not a real Jira extract. |
| `billable_hours_ratio` | float | Billable or productive-hour ratio | proportion 0–1 | 0.40–0.98 | Utilization proxy. |

---

## Learning signals (used in the label)

| Column | Type | Meaning | Units | Permitted range | Synthetic assumption |
| --- | --- | --- | --- | --- | --- |
| `training_hours` | float | Formal learning hours in the cycle | hours | 0.0–80.0 | Mildly related to latent ability plus noise. |
| `certifications_count` | integer | Active certifications | count | 0–6 | Count of simulated credentials. |
| `internal_hackathons_participated` | integer | Internal hackathons joined | count | 0–4 | Engagement / learning proxy. |

---

## Feedback signals (used in the label)

| Column | Type | Meaning | Units | Permitted range | Synthetic assumption |
| --- | --- | --- | --- | --- | --- |
| `peer_feedback_score` | float | Aggregated peer rating | 1–5 scale | 1.0–5.0 | Correlated with ability and cycle noise. |
| `previous_manager_score` | float | Manager rating from the **prior** cycle | 1–5 scale | 1.0–5.0 | Allowed as a lagged signal. It is **not** the next-cycle label. |
| `kudos_count` | integer | Recognition notes received | count | 0–20 | Informal recognition proxy. |

---

## Columns generated but **excluded** from the label

These exist so later EDA can show why some HR fields are risky or weakly related. They must **not** be treated as ground-truth drivers of performance.

| Column | Type | Meaning | Units | Permitted range | Why excluded from the label |
| --- | --- | --- | --- | --- | --- |
| `sick_days` | integer | Recorded sick leave | days | 0–15 | Health-related; must not define “performance.” Drawn independently of the label. |
| `unplanned_absences` | integer | Unplanned days away | days | 0–8 | Can proxy health or caregiving. Independent of the label. |
| `avg_login_hours` | float | Average daily system login time | hours/day | 5.0–11.0 | Surveillance-style activity; independent of the label. |

---

## Target

| Column | Type | Meaning | Values | Leakage rule |
| --- | --- | --- | --- | --- |
| `perf_band_next` | category | Simulated **next-cycle** performance band | `High`, `Medium`, `Low` | **Target only.** Do not use this column, or any raw score used to create it, as a model feature. |

Prediction horizon: given features observed in the current review cycle, estimate the band assigned for the **following** cycle. In this simulation the following-cycle band is generated from current-cycle work, learning, and feedback signals plus noise.

---

## Target-generation method

A continuous score in `[0, 1]` is a weighted sum of **normalized** work, training, and feedback signals. Weights are defined in `SCORE_WEIGHTS` inside `src/generate_data.py` and sum to 1.0:

| Normalized component | Weight | Source column |
| --- | --- | --- |
| `on_time_delivery_rate` | 0.16 | `on_time_delivery_rate` (already 0–1) |
| `delay_inverted` | 0.10 | `1 - scaled(avg_task_delay_days)` |
| `story_points_norm` | 0.10 | scaled `story_points_completed` |
| `bug_inverted` | 0.08 | `1 - scaled(bug_count)` |
| `code_review_norm` | 0.10 | scaled `code_review_score` |
| `billable_hours_ratio` | 0.07 | `billable_hours_ratio` |
| `projects_norm` | 0.05 | scaled `projects_count` |
| `training_norm` | 0.07 | scaled `training_hours` |
| `certifications_norm` | 0.04 | scaled `certifications_count` |
| `hackathons_norm` | 0.03 | scaled `internal_hackathons_participated` |
| `peer_feedback_norm` | 0.10 | scaled `peer_feedback_score` |
| `previous_manager_norm` | 0.07 | scaled `previous_manager_score` |
| `kudos_norm` | 0.03 | scaled `kudos_count` |

Then:

1. Add Gaussian noise: `N(0, 0.09)` (`NOISE_STD`).
2. Clip to `[0, 1]`.
3. Map to bands with fixed cutoffs:
   - `High` if score ≥ 0.63
   - `Medium` if 0.49 ≤ score < 0.63
   - `Low` if score < 0.49

Noise and the shared latent factor create **class overlap**: a High-band row can have mixed metrics, and a Low-band row is not a perfect inverse of High. Labels are therefore not trivially separable.

The following are **not** inputs to the score: `employee_id`, `review_cycle`, `department`, `job_level`, `education`, `experience_years`, `manager_tenure_years`, `sick_days`, `unplanned_absences`, `avg_login_hours`, `promotions_in_2y`, `salary_percentile_band`, and `perf_band_next` itself.

The numeric score is **not written** to the CSV, so models cannot train on the direct target-derived score.

---

## What this dataset is not

- Not production HR data.
- Not a validated predictor of job performance, potential, or “culture fit.”
- Not suitable for automated promotion, termination, pay, or disciplinary decisions.
