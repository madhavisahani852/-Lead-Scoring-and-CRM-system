"""
backend/seed_database.py
Idempotent seeder: loads cleaned_leads.csv, scores all rows using existing ML model,
then pushes to Supabase or seeds local in-memory store.
Run once: python -m backend.seed_database
"""
import sys
import math
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.predict import predict_lead_scores
from backend.supabase_db import (
    is_supabase_configured, seed_records,
    _clean, _sanitize, _now, _sb_post, _sb_get, initialize_local_store,
    LOCAL_LEADS
)


def _category(prob: float) -> str:
    if prob >= 0.75:
        return "HOT"
    if prob >= 0.40:
        return "WARM"
    return "COLD"


def _read_and_score() -> list[dict]:
    cleaned_path = PROJECT_ROOT / "cleaned_leads.csv"
    print(f"[Seeder] Reading {cleaned_path}")
    raw_df = pd.read_csv(cleaned_path)
    print(f"[Seeder] Loaded {len(raw_df)} rows. Running ML scoring...")
    scored_df = predict_lead_scores(raw_df)
    print("[Seeder] ML scoring complete.")

    records = []
    for i, row in scored_df.iterrows():
        lead_id = str(row["lead_id"]) if not pd.isna(row.get("lead_id", None)) else f"LEAD-{i:04d}"
        prob = _clean(row.get("conversion_probability"), 0.5)
        score = _clean(row.get("lead_score"), int(round(prob * 100)))
        category = _category(prob)
        priority = str(row.get("priority", "Medium"))
        target = row.get("target")
        converted = None
        outcome = None
        if not (isinstance(target, float) and math.isnan(target)) and target is not None:
            converted = bool(int(target))
            outcome = "Won" if converted else "Lost"

        def safe_str(col):
            v = row.get(col)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return ""
            return str(v)

        def safe_float(col, default=0.0):
            return _clean(row.get(col), default)

        records.append(_sanitize({
            "id": lead_id,
            "name": safe_str("name") or f"Lead {lead_id}",
            "email": safe_str("email") or f"{lead_id.lower()}@example.com",
            "phone": safe_str("phone"),
            "company": safe_str("company"),
            "job_title": safe_str("job_title"),
            "industry": safe_str("industry") or "SaaS",
            "company_size": safe_float("company_size", 100.0),
            "location": safe_str("location"),
            "lead_source": safe_str("lead_source") or "Website",
            "campaign": safe_str("campaign"),
            "product_interest": safe_str("product_interest"),
            "budget_range": safe_str("budget_range") or "Unknown",
            "website_visits": safe_float("website_visits"),
            "page_views": safe_float("page_views"),
            "pricing_page_visits": safe_float("pricing_page_visits"),
            "demo_requested": safe_str("demo_requested") or "No",
            "email_opens": safe_float("email_opens"),
            "form_completions": safe_float("form_completions"),
            "content_downloads": safe_float("content_downloads"),
            "previous_interactions": safe_float("previous_interactions"),
            "response_time_hours": safe_float("response_time_hours"),
            "num_calls": safe_float("num_calls"),
            "num_meetings": safe_float("num_meetings"),
            "score": score,
            "conversion_probability": round(prob, 4),
            "category": category,
            "priority": priority,
            "status": "New",
            "pipeline_stage": "New",
            "assigned_to": None,
            "tags": [],
            "converted": converted,
            "outcome": outcome,
            "created_at": safe_str("created_date") or "2024-01-01",
            "updated_at": safe_str("last_activity_date") or "2024-01-01",
        }))

    print(f"[Seeder] Prepared {len(records)} records.")
    return records


def seed_supabase(records: list[dict]):
    """Upload records to Supabase in batches. Skips existing IDs."""
    existing = _sb_get("leads", "&select=id&limit=2000") or []
    existing_ids = {str(r["id"]) for r in existing}

    print(f"[Seeder] Supabase: {len(existing_ids)} leads already present.")
    batch_size = 50
    new_records = [r for r in records if r["id"] not in existing_ids]
    print(f"[Seeder] Uploading {len(new_records)} new records in batches of {batch_size}...")

    imported = 0
    for i in range(0, len(new_records), batch_size):
        batch = new_records[i:i + batch_size]
        result = _sb_post("leads", batch)
        if result is not None:
            imported += len(batch)
        else:
            # Individual fallback
            for rec in batch:
                res = _sb_post("leads", rec)
                if res is not None:
                    imported += 1
        if i % 200 == 0 and i > 0:
            print(f"[Seeder]  ... {imported} uploaded so far")

    print(f"[Seeder] Done. {imported}/{len(new_records)} new records imported.")
    return imported


def main():
    records = _read_and_score()

    if is_supabase_configured():
        print("[Seeder] Supabase is configured. Seeding Supabase...")
        seed_supabase(records)
    else:
        print("[Seeder] Supabase not configured. Seeding local in-memory store only.")

    # Always seed in-memory store
    seed_records(records)
    print(f"[Seeder] In-memory store seeded with {len(records)} records.")


if __name__ == "__main__":
    main()
