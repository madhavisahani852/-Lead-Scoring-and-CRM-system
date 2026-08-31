"""
02_clean_data.py
----------------
Reproducible data-cleaning pipeline for the Lead Scoring & CRM system.

Input:
    raw_leads.csv

Outputs:
    cleaned_leads.csv
    quality_metrics.json

Cleaning steps:
1. Remove completely blank rows
2. Remove exact duplicates
3. Remove near-duplicate leads
4. Handle invalid numeric values
5. Fix impossible activity dates
6. Normalize lead-source categories
7. Normalize demo_requested
8. Normalize budget_range
9. Validate email addresses
10. Handle unresolved targets
11. Exclude leakage-prone columns
12. Generate quality metrics
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RAW_PATH = BASE_DIR / "raw_leads.csv"
CLEAN_PATH = BASE_DIR / "cleaned_leads.csv"
METRICS_PATH = BASE_DIR / "quality_metrics.json"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    """
    Remove leading/trailing spaces.
    Convert empty values to NaN.
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value == "":
        return np.nan

    return value


def normalize_email(value):
    """
    Normalize email for duplicate detection.
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    if value == "":
        return np.nan

    return value


def is_valid_email(value):
    """
    Basic email validation.
    """
    if pd.isna(value):
        return False

    value = str(value).strip()

    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

    return bool(re.match(pattern, value))


def normalize_demo(value):
    """
    Normalize demo_requested values.
    """

    if pd.isna(value):
        return "Unknown"

    value = str(value).strip().lower()

    if value in ["yes", "y"]:
        return "Yes"

    if value in ["no", "n"]:
        return "No"

    return "Unknown"


def normalize_budget(value):
    """
    Normalize budget range categories.
    """

    if pd.isna(value):
        return "Unknown"

    value = str(value).strip().lower()

    budget_mapping = {
        "<10k": "<10k",
        "10k-50k": "10k-50k",
        "50k-1l": "50k-1L",
        "50k-1l": "50k-1L",
        "1l-5l": "1L-5L",
        "5l+": "5L+",
        "10000-50000": "10k-50k",
    }

    return budget_mapping.get(value, "Unknown")


def normalize_lead_source(value):
    """
    Normalize lead-source categories.
    """

    if pd.isna(value):
        return "Unknown"

    value = str(value).strip().lower()

    lead_source_mapping = {
        "website": "Website",
        "web site": "Website",
        "web": "Website",

        "organic search": "Organic Search",
        "organic": "Organic Search",
        "seo": "Organic Search",

        "paid ads": "Paid Ads",
        "ads": "Paid Ads",
        "google ads": "Paid Ads",
        "ppc": "Paid Ads",

        "referral": "Referral",
        "referal": "Referral",

        "social media": "Social Media",
        "social": "Social Media",
        "fb/insta": "Social Media",

        "email campaign": "Email Campaign",
        "email": "Email Campaign",
    }

    return lead_source_mapping.get(value, "Unknown")


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

if not RAW_PATH.exists():
    raise FileNotFoundError(
        f"Could not find input file: {RAW_PATH}"
    )

df = pd.read_csv(RAW_PATH)

raw_row_count = len(df)
raw_col_count = len(df.columns)


# ============================================================
# 2. BASIC TEXT CLEANING
# ============================================================

# Handles both object and string columns without
# the pandas select_dtypes warning.

text_columns = df.select_dtypes(
    include=["object", "string"]
).columns

for col in text_columns:
    df[col] = df[col].apply(clean_text)


# ============================================================
# 3. REMOVE COMPLETELY BLANK ROWS
# ============================================================

blank_check_columns = [
    col for col in df.columns
    if col != "lead_id"
]

blank_row_mask = (
    df[blank_check_columns]
    .isna()
    .all(axis=1)
)

fully_blank_rows_removed = int(
    blank_row_mask.sum()
)

df = df.loc[
    ~blank_row_mask
].copy()


# ============================================================
# 4. REMOVE EXACT DUPLICATES
# ============================================================

# lead_id is excluded because duplicate records may have
# different lead IDs.

duplicate_columns = [
    col for col in df.columns
    if col != "lead_id"
]

exact_duplicate_mask = df.duplicated(
    subset=duplicate_columns,
    keep="first"
)

exact_duplicate_rows_removed = int(
    exact_duplicate_mask.sum()
)

df = df.loc[
    ~exact_duplicate_mask
].copy()


# ============================================================
# 5. NORMALIZE EMAIL FOR NEAR-DUPLICATE DETECTION
# ============================================================

df["email_norm"] = (
    df["email"]
    .apply(normalize_email)
)


# ============================================================
# 6. CONVERT DATE COLUMNS
# ============================================================

df["created_date"] = pd.to_datetime(
    df["created_date"],
    errors="coerce"
)

df["last_activity_date"] = pd.to_datetime(
    df["last_activity_date"],
    errors="coerce"
)


# ============================================================
# 7. REMOVE NEAR DUPLICATES
# ============================================================

# Near duplicate definition:
# Same normalized email + same company.
#
# Keep the earliest created lead.

df = df.sort_values(
    by=["created_date", "lead_id"],
    ascending=[True, True]
).copy()

near_duplicate_mask = (
    df["email_norm"].notna()
    & df["company"].notna()
    & df.duplicated(
        subset=["email_norm", "company"],
        keep="first"
    )
)

near_duplicate_rows_removed = int(
    near_duplicate_mask.sum()
)

df = df.loc[
    ~near_duplicate_mask
].copy()


# ============================================================
# 8. NUMERIC DATA CLEANING
# ============================================================

numeric_columns = [
    "company_size",
    "website_visits",
    "page_views",
    "pricing_page_visits",
    "email_opens",
    "form_completions",
    "content_downloads",
    "previous_interactions",
    "response_time_hours",
    "num_calls",
    "num_meetings",
]

# Convert numeric columns safely.
for col in numeric_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# Company size must be positive.

company_size_invalid = (
    df["company_size"].notna()
    & (df["company_size"] <= 0)
)

company_size_invalid_count = int(
    company_size_invalid.sum()
)

df.loc[
    company_size_invalid,
    "company_size"
] = np.nan


# Other numeric columns cannot be negative.

invalid_numeric_values_nulled = {
    "company_size_non_positive":
        company_size_invalid_count
}

non_negative_columns = [
    "website_visits",
    "page_views",
    "pricing_page_visits",
    "email_opens",
    "form_completions",
    "content_downloads",
    "previous_interactions",
    "response_time_hours",
    "num_calls",
    "num_meetings",
]

for col in non_negative_columns:

    invalid_mask = (
        df[col].notna()
        & (df[col] < 0)
    )

    invalid_count = int(
        invalid_mask.sum()
    )

    invalid_numeric_values_nulled[
        f"{col}_negative"
    ] = invalid_count

    df.loc[
        invalid_mask,
        col
    ] = np.nan


# ============================================================
# 9. FIX IMPOSSIBLE DATES
# ============================================================

impossible_date_mask = (
    df["last_activity_date"].notna()
    & df["created_date"].notna()
    & (
        df["last_activity_date"]
        < df["created_date"]
    )
)

impossible_last_activity_before_created = int(
    impossible_date_mask.sum()
)

df.loc[
    impossible_date_mask,
    "last_activity_date"
] = df.loc[
    impossible_date_mask,
    "created_date"
]


# ============================================================
# 10. NORMALIZE LEAD SOURCE
# ============================================================

lead_source_raw_variants = sorted(
    df["lead_source"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

df["lead_source"] = (
    df["lead_source"]
    .apply(normalize_lead_source)
)

lead_source_unmapped_values = sorted(
    df.loc[
        df["lead_source"] == "Unknown",
        "lead_source"
    ]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


# ============================================================
# 11. NORMALIZE DEMO REQUESTED
# ============================================================

df["demo_requested"] = (
    df["demo_requested"]
    .apply(normalize_demo)
)


# ============================================================
# 12. NORMALIZE BUDGET RANGE
# ============================================================

df["budget_range"] = (
    df["budget_range"]
    .apply(normalize_budget)
)

budget_values_mapped_to_unknown = int(
    (
        df["budget_range"] == "Unknown"
    ).sum()
)


# ============================================================
# 13. EMAIL VALIDATION
# ============================================================

df["email_valid"] = (
    df["email"]
    .apply(is_valid_email)
)

malformed_emails_flagged = int(
    (~df["email_valid"]).sum()
)


# ============================================================
# 14. CREATE TARGET / RESOLUTION FLAGS
# ============================================================

converted_numeric = pd.to_numeric(
    df["converted"],
    errors="coerce"
)

df["is_resolved"] = (
    df["status"].isin(
        ["Converted", "Lost"]
    )
    & converted_numeric.notna()
)

df["target"] = np.where(
    df["is_resolved"],
    converted_numeric,
    np.nan
)

unresolved_leads_excluded_from_target = int(
    (~df["is_resolved"]).sum()
)


# ============================================================
# 15. EXCLUDE LEAKAGE-PRONE COLUMNS
# ============================================================

# These columns should not be used as predictive features
# because they contain information that can leak the target.

leakage_columns = [
    "status",
    "converted",
    "sales_notes",
]


# ============================================================
# 16. REMOVE INTERMEDIATE COLUMNS
# ============================================================

# email_norm was only required for duplicate detection.

df = df.drop(
    columns=[
        "email_norm",
        *leakage_columns,
    ],
    errors="ignore"
)


# ============================================================
# 17. FINAL COLUMN SELECTION
# ============================================================

# IMPORTANT:
# Exactly 29 columns are intentionally retained.

final_columns = [
    "lead_id",
    "name",
    "email",
    "email_valid",
    "phone",
    "company",
    "industry",
    "company_size",
    "job_title",
    "location",
    "lead_source",
    "campaign",
    "product_interest",
    "budget_range",
    "website_visits",
    "page_views",
    "pricing_page_visits",
    "demo_requested",
    "email_opens",
    "form_completions",
    "content_downloads",
    "previous_interactions",
    "response_time_hours",
    "num_calls",
    "num_meetings",
    "created_date",
    "last_activity_date",
    "is_resolved",
    "target",
]

# Safety check
missing_final_columns = [
    col for col in final_columns
    if col not in df.columns
]

if missing_final_columns:

    raise ValueError(
        "The following required final columns "
        f"are missing: {missing_final_columns}"
    )


df = df[final_columns].copy()


# ============================================================
# 18. FORMAT DATES
# ============================================================

df["created_date"] = (
    pd.to_datetime(
        df["created_date"],
        errors="coerce"
    )
    .dt.strftime("%Y-%m-%d")
)

df["last_activity_date"] = (
    pd.to_datetime(
        df["last_activity_date"],
        errors="coerce"
    )
    .dt.strftime("%Y-%m-%d")
)


# ============================================================
# 19. FINAL DATA VALIDATION
# ============================================================

final_row_count = len(df)
final_col_count = len(df.columns)


if final_row_count != 1200:

    raise ValueError(
        f"Unexpected final row count: "
        f"{final_row_count}. Expected 1200."
    )


if final_col_count != 29:

    raise ValueError(
        f"Unexpected final column count: "
        f"{final_col_count}. Expected 29."
    )


# ============================================================
# 20. RESOLVED / UNRESOLVED COUNTS
# ============================================================

resolved_rows = int(
    df["is_resolved"].sum()
)

unresolved_rows = int(
    (~df["is_resolved"]).sum()
)


# ============================================================
# 21. CLASS BALANCE
# ============================================================

resolved_df = df[
    df["is_resolved"]
].copy()

class_balance = {}

if len(resolved_df) > 0:

    proportions = (
        resolved_df["target"]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    class_balance = {
        str(int(k)): round(
            float(v),
            4
        )
        for k, v in proportions.items()
    }


# ============================================================
# 22. MISSING VALUE METRICS
# ============================================================

missing_fraction_by_column = (
    df.isna()
    .mean()
    .round(4)
    .to_dict()
)


# ============================================================
# 23. QUALITY METRICS
# ============================================================

quality_metrics = {

    "raw_row_count":
        raw_row_count,

    "raw_col_count":
        raw_col_count,

    "fully_blank_rows_removed":
        fully_blank_rows_removed,

    "exact_duplicate_rows_removed":
        exact_duplicate_rows_removed,

    "near_duplicate_rows_removed":
        near_duplicate_rows_removed,

    "final_row_count":
        final_row_count,

    "final_col_count":
        final_col_count,

    "resolved_rows_for_training":
        resolved_rows,

    "unresolved_rows_scoring_only":
        unresolved_rows,

    "lead_source_raw_variants_found":
        lead_source_raw_variants,

    "lead_source_unmapped_values":
        lead_source_unmapped_values,

    "budget_values_mapped_to_unknown":
        budget_values_mapped_to_unknown,

    "invalid_numeric_values_nulled":
        invalid_numeric_values_nulled,

    "impossible_last_activity_before_created":
        impossible_last_activity_before_created,

    "malformed_emails_flagged":
        malformed_emails_flagged,

    "unresolved_leads_excluded_from_target":
        unresolved_leads_excluded_from_target,

    "missing_fraction_by_column":
        missing_fraction_by_column,

    "excluded_leakage_columns":
        leakage_columns,

    "class_balance_among_resolved":
        class_balance,
}


# ============================================================
# 24. SAVE CLEANED DATASET
# ============================================================

df.to_csv(
    CLEAN_PATH,
    index=False
)


# ============================================================
# 25. SAVE QUALITY METRICS
# ============================================================

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        quality_metrics,
        file,
        indent=2,
        default=str
    )


# ============================================================
# 26. PRINT FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("DATA CLEANING COMPLETED")
print("=" * 60)

print()
print(f"Raw rows       : {raw_row_count}")
print(f"Final rows     : {final_row_count}")
print(f"Raw columns    : {raw_col_count}")
print(f"Final columns  : {final_col_count}")

print()
print("Rows removed:")

print(
    f"  Blank rows       : "
    f"{fully_blank_rows_removed}"
)

print(
    f"  Exact duplicates : "
    f"{exact_duplicate_rows_removed}"
)

print(
    f"  Near duplicates  : "
    f"{near_duplicate_rows_removed}"
)

print()
print("Target:")

print(
    f"  Resolved         : "
    f"{resolved_rows}"
)

print(
    f"  Unresolved       : "
    f"{unresolved_rows}"
)

print()
print("Output files:")

print(
    f"  Created: {CLEAN_PATH}"
)

print(
    f"  Created: {METRICS_PATH}"
)

print()
print("Validation:")

print(
    f"  Dataset shape    : "
    f"({final_row_count}, {final_col_count})"
)

print(
    "  Expected shape   : (1200, 29)"
)

print()
print("=" * 60)