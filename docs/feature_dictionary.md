# Feature Dictionary

This document details all machine learning features utilized in the Lead Scoring & CRM Intelligence pipeline. `ml/config/feature_config.py` serves as the authoritative source of truth.

---

## Target Variable

| Feature Name | Description | Data Type | Source | Allowed Values | Used By |
|---|---|---|---|---|---|
| `target` | Binary indicator whether lead converted into a paying customer | Integer / Binary | CRM Outcome | `0` (Lost/Not Converted), `1` (Converted/Won), `NaN` (Unresolved) | Supervised Training & Evaluation |

---

## Numerical Features (`NUMERIC_COLS`)

| Feature Name | Description | Data Type | Source | Transformation | Allowed Values |
|---|---|---|---|---|---|
| `company_size` | Number of employees at lead company | Float / Int | CRM Profile | Median Imputation + StandardScaler | `≥ 1` |
| `website_visits` | Total website visits by lead | Int | Web Analytics | Median Imputation + StandardScaler | `≥ 0` |
| `page_views` | Total page views across website | Int | Web Analytics | Median Imputation + StandardScaler | `≥ 0` |
| `pricing_page_visits` | Visits to pricing & plans page | Int | Web Analytics | Median Imputation + StandardScaler | `≥ 0` |
| `email_opens` | Count of marketing emails opened | Int | Email Gateway | Median Imputation + StandardScaler | `≥ 0` |
| `form_completions` | Count of web form submissions | Int | Web Analytics | Median Imputation + StandardScaler | `≥ 0` |
| `content_downloads` | Downloads of whitepapers/e-books | Int | Content Portal | Median Imputation + StandardScaler | `≥ 0` |
| `previous_interactions` | Historical touchpoints before lead creation | Int | CRM History | Median Imputation + StandardScaler | `≥ 0` |
| `response_time_hours` | Average sales response time in hours | Float | CRM Activity | Median Imputation + StandardScaler | `≥ 0.0` |
| `num_calls` | Phone call interactions logged | Int | Call Logs | Median Imputation + StandardScaler | `≥ 0` |
| `num_meetings` | Completed sales meetings logged | Int | Calendar / CRM | Median Imputation + StandardScaler | `≥ 0` |

---

## Categorical / Nominal Features (`NOMINAL_COLS`)

| Feature Name | Description | Data Type | Source | Transformation | Allowed Values |
|---|---|---|---|---|---|
| `industry` | Primary industry sector of lead company | String | CRM Input | Constant Imputer ("Unknown") + OneHotEncoder | SaaS, E-commerce, Healthcare, Finance, Retail, etc. |
| `lead_source` | Channel through which lead was acquired | String | Marketing Attribution | Constant Imputer ("Unknown") + OneHotEncoder | Website, Paid Ads, Referral, Organic Search, etc. |
| `product_interest` | Core product module interested in | String | Lead Form | Constant Imputer ("Unknown") + OneHotEncoder | Core Platform, Enterprise, Add-on Module, etc. |
| `demo_requested` | Whether product demo was requested | String | Lead Form | Constant Imputer ("Unknown") + OneHotEncoder | Yes, No, Unknown |

---

## Ordinal Features (`ORDINAL_COLS`)

| Feature Name | Description | Data Type | Source | Transformation / Mapping | Allowed Values |
|---|---|---|---|---|---|
| `budget_range` | Self-reported company budget bracket | String → Int | Lead Form | OrdinalMapper: `Unknown`: 0, `<10k`: 1, `10k-50k`: 2, `50k-1L`: 3, `1L-5L`: 4, `5L+`: 5 | Unknown, <10k, 10k-50k, 50k-1L, 1L-5L, 5L+ |

---

## Engineered Features (`FeatureEngineer`)

| Feature Name | Description | Data Type | Formula / Logic |
|---|---|---|---|
| `total_engagement_score` | Composite score of all lead interactions | Float | `sum(website_visits + page_views + pricing_page_visits + demo_requested + email_opens + content_downloads)` |
| `pricing_intent_ratio` | Ratio of pricing visits to overall visits | Float | `pricing_page_visits / website_visits` (0.0 if visits == 0) |
| `days_since_last_activity` | Recency measure of lead engagement | Int | `(as_of_date - last_activity_date).days` (999 if null) |

---

## Excluded / Dropped Columns (`DROP_COLS`)

| Feature Name | Reason for Exclusion |
|---|---|
| `lead_id`, `name`, `email`, `phone`, `company` | Personal identifiers / Non-generalizable unique metadata |
| `campaign`, `job_title`, `location` | Excluded per feature configuration |
| `created_date`, `last_activity_date` | Raw timestamp strings (transformed into recency feature) |
