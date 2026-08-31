# Data Dictionary — Lead Scoring & CRM Intelligence Tool

Project: Task A1 — Dataset Engineering
Owner: Samiksha
Files covered: `raw_leads.csv` (1,250 rows, immutable) and `cleaned_leads.csv` (1,200 rows, analysis-ready)

Field names below are as they appear in **cleaned_leads.csv**. Where the raw
column name or raw representation differs, it's noted in the "Raw source" column.

## 1. Identifiers & firmographic fields

| Field | Type | Meaning | Raw source | Missingness | Allowed values |
|---|---|---|---|---|---|
| `lead_id` | string | Unique lead identifier assigned at capture | `lead_id` (unchanged) | 0% | `L#####`, unique |
| `name` | string | Lead's full name | `name` (unchanged) | ~0.9% | free text |
| `email` | string | Lead's email address as captured | `email` (unchanged) | ~8.8% | free text; validity flagged separately |
| `email_valid` | boolean | Whether `email` contains an `@` and is structurally plausible | derived from `email` | 0% | `True` / `False` |
| `phone` | string | Contact phone number | `phone` (unchanged) | ~7.3% | free text, `+91-` prefixed |
| `company` | string | Lead's company/organization name | `company` (unchanged) | 0% | free text |
| `industry` | string | Company's industry vertical | `industry` (unchanged) | ~3.2% | SaaS, E-commerce, Manufacturing, Healthcare, Education, Finance, Retail, Logistics, Real Estate, Media |
| `company_size` | float | Number of employees at lead's company | `company_size`, invalid entries (≤0) nulled | ~3.3% (incl. nulled invalids) | positive integer |
| `job_title` | string | Lead's job title | `job_title` (unchanged) | 0% | free text |
| `location` | string | Lead's city / region | `location` (unchanged) | 0% | free text |

## 2. Acquisition fields

| Field | Type | Meaning | Raw source | Missingness | Allowed values |
|---|---|---|---|---|---|
| `lead_source` | string | Standardized acquisition channel | `lead_source`, mapped from 24 raw spelling/casing variants (see quality report) | 0% (unmapped → `Unknown`) | Website, Organic Search, Paid Ads, Referral, Social Media, Email Campaign, Unknown |
| `campaign` | string | Campaign code, only populated for Paid Ads / Email Campaign sources | `campaign` (unchanged) | ~65.9% (expected — only paid/email sources have one) | `CMP-###` or blank |
| `product_interest` | string | Product/plan the lead expressed interest in | `product_interest` (unchanged) | 0% | Starter Plan, Growth Plan, Enterprise Plan, Add-on Module |
| `budget_range` | string | Lead's stated budget bracket | `budget_range`, non-standard tokens (e.g. `n/a`, `unknown`, `10000-50000`) mapped to canonical buckets or `Unknown` | 0% (non-standard → `Unknown`) | <10k, 10k-50k, 50k-1L, 1L-5L, 5L+, Unknown |

## 3. Engagement fields (behavioral / trainable features)

| Field | Type | Meaning | Raw source | Missingness | Allowed values |
|---|---|---|---|---|---|
| `website_visits` | float | Total website visits by the lead | `website_visits`, negative values nulled | ~1.1% | ≥ 0 |
| `page_views` | float | Total page views across visits | `page_views`, negative values nulled | ~0.6% | ≥ 0 |
| `pricing_page_visits` | float | Number of visits to the pricing page | `pricing_page_visits` (unchanged) | 0% | ≥ 0 |
| `demo_requested` | string | Whether the lead requested a product demo | `demo_requested`, casing/abbreviation variants (`yes`/`Y`/`no`/`N`) standardized | 0% (unmapped → `Unknown`) | Yes, No, Unknown |
| `email_opens` | float | Number of marketing emails opened | `email_opens` (unchanged) | 0% | ≥ 0 |
| `form_completions` | float | Number of on-site forms completed | `form_completions` (unchanged) | 0% | ≥ 0 |
| `content_downloads` | float | Number of gated content assets downloaded | `content_downloads` (unchanged) | 0% | ≥ 0 |

## 4. Sales activity fields

| Field | Type | Meaning | Raw source | Missingness | Allowed values |
|---|---|---|---|---|---|
| `previous_interactions` | float | Count of prior sales touches (calls/emails/meetings combined) | `previous_interactions` (unchanged) | 0% | ≥ 0 |
| `response_time_hours` | float | Hours between lead creation and first sales response | `response_time_hours` (unchanged, genuinely missing where no response yet) | ~9.4% | ≥ 0 |
| `num_calls` | float | Number of sales calls made to this lead | `num_calls` (unchanged) | 0% | ≥ 0 |
| `num_meetings` | float | Number of meetings held with this lead | `num_meetings` (unchanged) | 0% | ≥ 0 |

## 5. Lifecycle & target fields

| Field | Type | Meaning | Raw source | Missingness | Allowed values |
|---|---|---|---|---|---|
| `created_date` | date (YYYY-MM-DD) | Date the lead entered the CRM | `created_date` (unchanged) | 0% | valid date |
| `last_activity_date` | date (YYYY-MM-DD) | Date of the most recent recorded activity | `last_activity_date`; 12 rows where this preceded `created_date` were corrected to equal `created_date` | 0% | valid date, ≥ `created_date` |
| `is_resolved` | boolean | Whether the lead has a final outcome (Converted or Lost) as opposed to still being in an open pipeline stage | derived from raw `status`/`converted` | 0% | `True` / `False` |
| `target` | float (0/1/NaN) | **Model target.** `1` = converted, `0` = not converted/lost, `NaN` = unresolved lead (excluded from training, eligible for live scoring only) | derived from raw `converted` per Task A1 target definition | ~15.3% (all unresolved leads, by design) | `1.0`, `0.0`, or null |

## 6. Fields present in `raw_leads.csv` but EXCLUDED from `cleaned_leads.csv`

| Field | Reason for exclusion |
|---|---|
| `status` | Redundant with `is_resolved` + `target`; raw pipeline-stage labels aren't needed for modeling and duplicating them risks confusion with the target. |
| `converted` | Superseded by cleaned `target` column (same information, standardized encoding). |
| `sales_notes` | **Leakage.** Free-text notes are written by sales reps *after* the outcome is known (e.g. "deal won", "marking lost") and directly encode the target. Including this as a feature would let a model "cheat" by reading the outcome instead of predicting it. Kept in the raw file for audit purposes only. |
| `email_norm` | Internal helper column used only for deduplication logic; not meaningful downstream. |

## 7. Fields considered but not available at scoring time (excluded from training features)

These fields exist conceptually in the PRD (Section 10) but are **not included**
in this dataset/feature set because they would not be reliably available the
moment a *new* lead needs to be scored, or they raise privacy concerns per the
PRD's Section 31/32 (avoid sensitive personal attributes):

- Any field derived from the eventual sales outcome (see leakage note above).
- Granular personal identifiers beyond `name`/`email`/`phone` (e.g. no age, gender, or other demographic attributes were collected — the PRD explicitly excludes sensitive personal attributes from scoring).
