# Data Quality Report — Lead Scoring & CRM Intelligence Tool

Project: Task A1 — Dataset Engineering
Owner: Samiksha
Generated from: `01_generate_raw_data.py` → `raw_leads.csv` → `02_clean_data.py` → `cleaned_leads.csv`
(numbers below are pulled directly from `quality_metrics.json`, produced by the cleaning run)

## 1. Overview

| | Raw | Cleaned |
|---|---|---|
| Rows | 1,250 | 1,200 |
| Columns | 29 | 29 (4 dropped, 5 renamed/derived — see data dictionary §6) |
| Rows usable for **training** (`is_resolved = True`) | — | 1,017 |
| Rows for **scoring only** (unresolved, `target` is null) | — | 183 |
| Class balance among resolved leads | — | Converted 60.7% / Lost 39.3% |

`raw_leads.csv` is never modified in place — all fixes are applied to a
working copy inside `02_clean_data.py`, so the raw export always remains
available for re-auditing or re-running the pipeline with different rules.

## 2. Duplicates

| Issue | Count | Treatment |
|---|---|---|
| Fully blank/garbage rows (corrupted export rows) | 5 | Dropped |
| Exact duplicate rows (identical across every field except `lead_id`) | 25 | Dropped, kept first occurrence |
| Near-duplicate leads (same email + company, re-submitted under a new `lead_id`, sometimes with re-cased email) | 20 | Dropped, kept the **earliest** `created_date` occurrence as the canonical record |

## 3. Invalid values

| Field | Issue | Count found | Treatment |
|---|---|---|---|
| `company_size` | Non-positive (0 or negative) — data entry error | 40 | Set to null rather than guessed |
| `website_visits` | Negative counts | 13 | Set to null |
| `page_views` | Negative counts | 7 | Set to null |
| `last_activity_date` | Earlier than `created_date` (logically impossible) | 12 | Corrected to equal `created_date` (conservative fix — treats it as "no activity yet recorded past creation" rather than dropping the row) |
| `email` | Malformed (missing `@`) | 23 | Flagged via new `email_valid = False` column rather than deleted or guessed at, so downstream users can decide whether to exclude these leads from email-based outreach features |

No negative values were found in `pricing_page_visits`, `email_opens`,
`form_completions`, `content_downloads`, `previous_interactions`,
`num_calls`, or `num_meetings` in this run.

## 4. Inconsistent categories

`lead_source` arrived in **24 different raw spellings/casings** for the same
6 underlying channels, for example:

- Website: `Website`, `website`, `WEBSITE`, `web site`, `Web`
- Organic Search: `Organic Search`, `Organic`, `organic search`, `SEO`
- Paid Ads: `Paid Ads`, `Ads`, `paid ads`, `Google Ads`, `PPC`
- Referral: `Referral`, `referral`, `Referal` (typo)
- Social Media: `Social Media`, `Social`, `social media`, `FB/Insta`
- Email Campaign: `Email Campaign`, `Email`, `email campaign`

All variants were mapped to 6 canonical labels; nothing was left unmapped in
this run (0 unmapped values), so no rows fell into the `Unknown` bucket for
`lead_source`.

`demo_requested` had mixed casing/abbreviations (`Yes/yes/Y`, `No/no/N`) —
standardized to `Yes` / `No`, with true blanks mapped to `Unknown`.

`budget_range` had free-text and placeholder values (`n/a`, `NA`, `unknown`,
`10000-50000`, blank) alongside the canonical brackets. **129 rows** did not
match a canonical bracket and were mapped to `Unknown` rather than dropped or
imputed, since guessing a budget would fabricate information that was never
actually captured.

## 5. Missing data

Missingness after cleaning, by field (fraction of 1,200 rows):

| Field | Missing % | Likely cause |
|---|---|---|
| `campaign` | 65.9% | Expected — only Paid Ads / Email Campaign leads have a campaign code |
| `converted` (raw) / `target` (cleaned) | 15.3% | Genuinely unresolved leads still in an open pipeline stage — see §6 |
| `response_time_hours` | 9.4% | No sales response recorded yet |
| `email` | 8.8% | Not collected at capture (e.g. phone-only inbound leads) |
| `phone` | 7.3% | Not collected at capture |
| `budget_range` | 0% (post-mapping; see §4) | Non-standard entries redirected to `Unknown` rather than left as nulls |
| `industry`, `company_size`, `demo_requested`, `website_visits`, `page_views`, `name` | 0.6%–4.4% | Sporadic data entry gaps |

No missing values were imputed with fabricated numbers. Nulls were left as
nulls (or as an explicit `Unknown` category for categorical fields) so that
modeling choices about imputation are made deliberately downstream, not
silently baked into this dataset.

## 6. Target definition & unresolved leads

Per the task requirement, the target is defined as:
`converted = 1`, `not converted / lost = 0`.

**183 leads (15.3%)** have no final outcome yet — they're still sitting in an
open pipeline stage (New / Contacted / Qualified / Demo Scheduled / Proposal).
These are **not** assigned `target = 0`, because that would falsely label an
open opportunity as a loss and bias the model against currently-active leads.
Instead:

- `is_resolved = False` and `target = NaN` for these rows.
- They are **excluded from model training** but **retained in the dataset**
  so they can still flow through the live scoring pipeline (the whole point
  of the product — scoring leads before their outcome is known).

## 7. Leakage detection

One field, `sales_notes`, was identified as a leakage risk: it is a
free-text note written by the sales rep *after* the deal outcome is known
(e.g. "Client signed the contract, deal won!" / "Budget cut, deal lost.").
Including it as a model feature would leak the answer directly into the
inputs. It has been **excluded from `cleaned_leads.csv`** and is retained
only in `raw_leads.csv` for audit/traceability.

The raw `status` and `converted` columns were also excluded from the
cleaned feature set (superseded by `is_resolved`/`target`) to avoid
duplicate/confusable representations of the outcome sitting next to the
features.

## 8. Reproducibility

Both scripts are deterministic (`numpy` RNG seeded at 42). Re-running
`01_generate_raw_data.py` followed by `02_clean_data.py` reproduces
identical `raw_leads.csv`, `cleaned_leads.csv`, and `quality_metrics.json`
files byte-for-byte, satisfying the "reproducible dataset" deliverable
requirement.
