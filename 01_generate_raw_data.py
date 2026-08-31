"""
01_generate_raw_data.py
------------------------
Generates a synthetic but realistic RAW historical CRM/lead dataset for the
Lead Scoring & CRM Intelligence Tool project (Task A1 - Dataset Engineering).

This script deliberately injects the kinds of problems real CRM exports have,
so the cleaning step (02_clean_data.py) has something genuine to fix:
  - exact duplicate rows
  - near-duplicate leads (same person, different lead_id, re-entered)
  - missing values in multiple fields (MCAR + a couple of MNAR patterns)
  - inconsistent category casing/spelling ("Website" vs "website" vs "web")
  - invalid / out-of-range values (negative visit counts, impossible dates,
    company_size = 0, budget_range typos)
  - a leakage-prone field (post-outcome "sales_notes_mentions_won") that must
    NOT be used as a training feature
  - unresolved leads (status still "New"/"Contacted"/"Qualified" with no
    final Converted/Lost outcome yet) mixed in with resolved ones

The script is deterministic (fixed random seed) so re-running it reproduces
the exact same raw file -> that raw file is then treated as immutable
("raw dataset preserved") and all cleaning happens on a copy.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)
N_LEADS = 1200  # base number of genuine leads before duplicates are injected

OUT_PATH = "raw_leads.csv"

# ---------------------------------------------------------------------------
# Reference / lookup data
# ---------------------------------------------------------------------------
INDUSTRIES = ["SaaS", "E-commerce", "Manufacturing", "Healthcare", "Education",
              "Finance", "Retail", "Logistics", "Real Estate", "Media"]

JOB_TITLES = ["Founder", "CEO", "Marketing Manager", "Sales Manager",
              "Operations Manager", "IT Manager", "Procurement Head",
              "Business Analyst", "Director", "VP Sales"]

LOCATIONS = ["Pune", "Mumbai", "Bangalore", "Delhi", "Hyderabad",
             "Chennai", "Ahmedabad", "Kolkata", "Jaipur", "Remote"]

# Intentionally inconsistent source labels (same real-world source, many spellings)
SOURCE_VARIANTS = {
    "Website": ["Website", "website", "WEBSITE", "web site", "Web"],
    "Organic Search": ["Organic Search", "Organic", "organic search", "SEO"],
    "Paid Ads": ["Paid Ads", "Ads", "paid ads", "Google Ads", "PPC"],
    "Referral": ["Referral", "referral", "Referal"],  # includes a typo
    "Social Media": ["Social Media", "Social", "social media", "FB/Insta"],
    "Email Campaign": ["Email Campaign", "Email", "email campaign"],
}

BUDGET_RANGES_CLEAN = ["<10k", "10k-50k", "50k-1L", "1L-5L", "5L+"]
BUDGET_RANGES_DIRTY = ["<10k", "10k-50k", "50k-1L", "1L-5L", "5L+",
                        "10000-50000", "n/a", "NA", "unknown", ""]

PRODUCTS = ["Starter Plan", "Growth Plan", "Enterprise Plan", "Add-on Module"]

STATUS_RESOLVED = ["Converted", "Lost"]
STATUS_UNRESOLVED = ["New", "Contacted", "Qualified", "Demo Scheduled", "Proposal"]

def rand_date(start, end):
    delta = end - start
    return start + timedelta(days=int(rng.integers(0, delta.days + 1)))

START = datetime(2024, 1, 1)
END = datetime(2026, 8, 1)

def make_email(name, company, i, dirty=False):
    if dirty and rng.random() < 0.5:
        return ""  # missing email
    base = name.lower().replace(" ", ".")
    dom = company.lower().replace(" ", "").replace(",", "")[:12] or "example"
    email = f"{base}{i}@{dom}.com"
    if dirty and rng.random() < 0.3:
        email = email.replace("@", "")  # malformed / invalid email
    return email

FIRST_NAMES = ["Rahul", "Priya", "Amit", "Sneha", "Vikram", "Anjali", "Rohan",
               "Neha", "Karan", "Divya", "Arjun", "Pooja", "Sanjay", "Kavya",
               "Rajesh", "Meera", "Suresh", "Ritu", "Manoj", "Isha"]
LAST_NAMES = ["Sharma", "Verma", "Patel", "Gupta", "Reddy", "Nair", "Iyer",
              "Singh", "Mehta", "Joshi", "Kulkarni", "Rao", "Das", "Kapoor"]
COMPANY_ROOTS = ["Technologies", "Solutions", "Systems", "Industries", "Labs",
                  "Enterprises", "Ventures", "Networks", "Software", "Global"]
COMPANY_NAMES_POOL = [f"{a} {b}" for a in
                       ["ABC", "XYZ", "Nova", "Bright", "Prime", "Zenith",
                        "Orbit", "Vertex", "Alpha", "Summit", "Blue", "Nimbus"]
                       for b in COMPANY_ROOTS]

def generate_lead(i):
    fn = rng.choice(FIRST_NAMES)
    ln = rng.choice(LAST_NAMES)
    name = f"{fn} {ln}"
    company = rng.choice(COMPANY_NAMES_POOL)
    industry = rng.choice(INDUSTRIES)
    job_title = rng.choice(JOB_TITLES)
    location = rng.choice(LOCATIONS)

    # company_size: mostly valid, occasionally invalid (0 or negative -> data entry error)
    company_size = int(rng.choice([5, 10, 25, 50, 100, 250, 500, 1000]))
    if rng.random() < 0.02:
        company_size = 0
    if rng.random() < 0.01:
        company_size = -10  # invalid

    # source: pick a "true" canonical source then corrupt the label
    canonical_source = rng.choice(list(SOURCE_VARIANTS.keys()))
    source_label = rng.choice(SOURCE_VARIANTS[canonical_source])

    campaign = f"CMP-{int(rng.integers(100, 999))}" if canonical_source in \
        ("Paid Ads", "Email Campaign") else ""

    product_interest = rng.choice(PRODUCTS)
    budget_range = rng.choice(BUDGET_RANGES_DIRTY, p=[0.2, 0.25, 0.2, 0.15, 0.05,
                                                        0.05, 0.03, 0.03, 0.02, 0.02])

    created = rand_date(START, END)
    # last_activity should be >= created, but inject a few impossible rows
    last_activity = created + timedelta(days=int(rng.integers(0, 120)))
    if rng.random() < 0.01:
        last_activity = created - timedelta(days=30)  # invalid: before creation

    website_visits = int(rng.poisson(4))
    if rng.random() < 0.01:
        website_visits = -3  # invalid negative count
    page_views = website_visits + int(rng.poisson(3))
    pricing_visits = int(rng.binomial(min(website_visits, 10) if website_visits > 0 else 0, 0.3))
    demo_requested = rng.choice(["Yes", "No", "yes", "no", "Y", "N", ""],
                                 p=[0.25, 0.55, 0.05, 0.05, 0.03, 0.03, 0.04])
    email_opens = int(rng.poisson(2))
    form_completions = int(rng.binomial(3, 0.4))
    content_downloads = int(rng.poisson(1))
    prev_interactions = int(rng.poisson(2))
    response_time_hours = round(float(rng.exponential(12)), 1) if rng.random() > 0.1 else np.nan
    num_calls = int(rng.poisson(1.5))
    num_meetings = int(rng.poisson(0.7))

    # Determine ground-truth "quality" signal to make the target semi-learnable
    quality_score = (
        (1 if demo_requested in ("Yes", "yes", "Y") else 0) * 2
        + (pricing_visits > 1) * 1.5
        + (company_size >= 50) * 1
        + (canonical_source in ("Referral", "Organic Search")) * 1
        + (num_meetings > 0) * 1.5
        - (response_time_hours if not np.isnan(response_time_hours) else 24) / 48
    )
    conv_prob = 1 / (1 + np.exp(-(quality_score - 2)))

    # status: some leads are still unresolved (no final outcome yet)
    if rng.random() < 0.18:
        status = rng.choice(STATUS_UNRESOLVED)
        converted = ""  # unresolved -> target not yet known
    else:
        converted = 1 if rng.random() < conv_prob else 0
        status = "Converted" if converted == 1 else "Lost"

    # Leakage field: a note field that literally mentions the outcome ->
    # must be EXCLUDED from training features, it's only knowable after the fact.
    if converted == 1:
        sales_notes = rng.choice(["Client signed the contract, deal won!",
                                   "Deal closed successfully.",
                                   "Great call, moving to won."])
    elif converted == 0:
        sales_notes = rng.choice(["Lead went cold, marking lost.",
                                   "Budget cut, deal lost.",
                                   "No response, closing as lost."])
    else:
        sales_notes = rng.choice(["Follow-up scheduled.", "Waiting on demo.",
                                   "", "Sent pricing info."])

    email = make_email(name, company, i, dirty=(rng.random() < 0.15))
    phone = f"+91-9{rng.integers(100000000, 999999999)}" if rng.random() > 0.08 else ""

    return {
        "lead_id": f"L{i:05d}",
        "name": name if rng.random() > 0.01 else "",  # rare missing name
        "email": email,
        "phone": phone,
        "company": company,
        "industry": industry if rng.random() > 0.03 else "",
        "company_size": company_size,
        "job_title": job_title,
        "location": location,
        "lead_source": source_label,
        "campaign": campaign,
        "product_interest": product_interest,
        "budget_range": budget_range,
        "website_visits": website_visits,
        "page_views": page_views,
        "pricing_page_visits": pricing_visits,
        "demo_requested": demo_requested,
        "email_opens": email_opens,
        "form_completions": form_completions,
        "content_downloads": content_downloads,
        "previous_interactions": prev_interactions,
        "response_time_hours": response_time_hours,
        "num_calls": num_calls,
        "num_meetings": num_meetings,
        "created_date": created.strftime("%Y-%m-%d"),
        "last_activity_date": last_activity.strftime("%Y-%m-%d"),
        "status": status,
        "sales_notes": sales_notes,   # LEAKAGE FIELD - keep in raw, exclude from features
        "converted": converted,       # TARGET (blank string = unresolved)
    }

rows = [generate_lead(i) for i in range(1, N_LEADS + 1)]
df = pd.DataFrame(rows)

# --- Inject duplicate rows (exact copies re-exported from CRM) ---
exact_dupe_idx = rng.choice(df.index, size=25, replace=False)
df = pd.concat([df, df.loc[exact_dupe_idx]], ignore_index=True)

# --- Inject near-duplicates: same person/company, new lead_id, re-submitted the form ---
near_dupe_source_idx = rng.choice(df.index, size=20, replace=False)
near_dupes = df.loc[near_dupe_source_idx].copy()
near_dupes["lead_id"] = [f"L{9000+i:05d}" for i in range(len(near_dupes))]
# slightly different created_date (re-submitted later) and possibly re-cased email
near_dupes["created_date"] = [
    (datetime.strptime(d, "%Y-%m-%d") + timedelta(days=int(rng.integers(1, 10)))).strftime("%Y-%m-%d")
    for d in near_dupes["created_date"]
]
near_dupes["email"] = near_dupes["email"].str.upper()
df = pd.concat([df, near_dupes], ignore_index=True)

# --- Inject a handful of fully blank/garbage rows (corrupted export) ---
garbage_rows = pd.DataFrame([{c: np.nan for c in df.columns} for _ in range(5)])
garbage_rows["lead_id"] = [f"L{9500+i:05d}" for i in range(5)]
df = pd.concat([df, garbage_rows], ignore_index=True)

# Shuffle rows to mimic a real export (duplicates not conveniently adjacent)
df = df.sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)

df.to_csv(OUT_PATH, index=False)
print(f"Wrote {len(df)} rows to {OUT_PATH}")
print(df.isna().sum())
