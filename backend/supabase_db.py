"""
backend/supabase_db.py
CRM Database Layer — Supabase REST API with local in-memory fallback.
Serves all 1,200 ML-scored leads from cleaned_leads.csv.
Sanitizes all NaN/Inf values for JSON compliance.
"""
import os
import uuid
import math
from datetime import datetime
from typing import Dict, Any, List, Optional
import requests
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

from ml.inference.predict import predict_lead_scores

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = (
    os.getenv("SUPABASE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
    or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_SERVICE_KEY", "")
).strip()

# In-memory CRM stores (used when Supabase is not configured)
LOCAL_LEADS: List[Dict[str, Any]] = []
LOCAL_NOTES: List[Dict[str, Any]] = []
LOCAL_ACTIVITIES: List[Dict[str, Any]] = []
IS_INITIALIZED = False

PIPELINE_STAGES = ["New", "Contacted", "Qualified", "Demo Scheduled",
                   "Proposal", "Negotiation", "Won", "Lost"]


# ── Utilities ─────────────────────────────────────────────────────────────────

def is_supabase_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _headers(prefer: str = "return=representation") -> Dict[str, str]:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": prefer,
    }


def _clean(val, default=None):
    """Convert NaN/Inf to JSON-safe values."""
    if val is None:
        return default
    try:
        if isinstance(val, (float, np.floating)):
            if math.isnan(val) or math.isinf(val):
                return default
            return float(val)
        if isinstance(val, (int, np.integer)):
            return int(val)
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
        if pd.isna(val):
            return default
    except Exception:
        pass
    return val


def _sanitize(r: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _clean(v) for k, v in r.items()}


def _now() -> str:
    return datetime.now().isoformat()


def _add_activity(lead_id: str, activity_type: str, description: str, actor: str = "System"):
    act = _sanitize({
        "id": str(uuid.uuid4()),
        "lead_id": lead_id,
        "activity_type": activity_type,
        "description": description,
        "actor": actor,
        "created_at": _now(),
    })
    if is_supabase_configured():
        try:
            requests.post(
                f"{SUPABASE_URL.rstrip('/')}/rest/v1/lead_activities",
                json=act, headers=_headers("return=minimal"), timeout=4
            )
        except Exception:
            LOCAL_ACTIVITIES.insert(0, act)
    else:
        LOCAL_ACTIVITIES.insert(0, act)


# ── Seed / Initialize ─────────────────────────────────────────────────────────

def seed_records(records: List[Dict[str, Any]]):
    """Called by seed_database.py to populate the in-memory store."""
    global LOCAL_LEADS, IS_INITIALIZED
    LOCAL_LEADS = [_sanitize(r) for r in records]
    IS_INITIALIZED = True


def _category_from_prob(prob: float) -> str:
    if prob >= 0.75:
        return "HOT"
    if prob >= 0.40:
        return "WARM"
    return "COLD"


def initialize_local_store():
    global LOCAL_LEADS, IS_INITIALIZED
    if IS_INITIALIZED and LOCAL_LEADS:
        return

    cleaned_path = PROJECT_ROOT / "cleaned_leads.csv"
    if not cleaned_path.exists():
        IS_INITIALIZED = True
        return

    raw_df = pd.read_csv(cleaned_path)
    scored_df = predict_lead_scores(raw_df)
    records = []

    for _, row in scored_df.iterrows():
        lead_id = str(row["lead_id"]) if not pd.isna(row.get("lead_id")) else f"LEAD-{_}"
        prob = _clean(row.get("conversion_probability"), 0.5)
        score = _clean(row.get("lead_score"), int(round(prob * 100)))
        category = _category_from_prob(prob)
        priority = str(row.get("priority", "Medium"))
        created_val = str(row.get("created_date")) if not pd.isna(row.get("created_date")) else "2024-01-01"
        target = row.get("target")
        converted = None
        outcome = None
        if not pd.isna(target):
            converted = bool(int(target))
            outcome = "Won" if converted else "Lost"

        rec = {
            "id": lead_id,
            "name": str(row["name"]) if not pd.isna(row.get("name")) else f"Lead {lead_id}",
            "email": str(row["email"]) if not pd.isna(row.get("email")) else f"{lead_id.lower()}@company.com",
            "phone": str(row["phone"]) if not pd.isna(row.get("phone")) else "",
            "company": str(row["company"]) if not pd.isna(row.get("company")) else "Unknown",
            "job_title": str(row["job_title"]) if not pd.isna(row.get("job_title")) else "",
            "industry": str(row["industry"]) if not pd.isna(row.get("industry")) else "SaaS",
            "company_size": _clean(row.get("company_size"), 100.0),
            "location": str(row["location"]) if not pd.isna(row.get("location")) else "",
            "lead_source": str(row["lead_source"]) if not pd.isna(row.get("lead_source")) else "Website",
            "campaign": str(row["campaign"]) if not pd.isna(row.get("campaign")) else "",
            "product_interest": str(row["product_interest"]) if not pd.isna(row.get("product_interest")) else "",
            "budget_range": str(row["budget_range"]) if not pd.isna(row.get("budget_range")) else "Unknown",
            "website_visits": _clean(row.get("website_visits"), 0.0),
            "page_views": _clean(row.get("page_views"), 0.0),
            "pricing_page_visits": _clean(row.get("pricing_page_visits"), 0.0),
            "demo_requested": str(row["demo_requested"]) if not pd.isna(row.get("demo_requested")) else "No",
            "email_opens": _clean(row.get("email_opens"), 0.0),
            "form_completions": _clean(row.get("form_completions"), 0.0),
            "content_downloads": _clean(row.get("content_downloads"), 0.0),
            "previous_interactions": _clean(row.get("previous_interactions"), 0.0),
            "response_time_hours": _clean(row.get("response_time_hours"), 0.0),
            "num_calls": _clean(row.get("num_calls"), 0.0),
            "num_meetings": _clean(row.get("num_meetings"), 0.0),
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
            "created_at": created_val,
            "updated_at": created_val,
        }
        records.append(_sanitize(rec))

    LOCAL_LEADS = records
    IS_INITIALIZED = True


# ── Supabase helpers ───────────────────────────────────────────────────────────

def _sb_get(table: str, filters: str = "") -> Optional[List[Dict]]:
    try:
        url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{table}?select=*{filters}"
        r = requests.get(url, headers=_headers(), timeout=5)
        if r.status_code == 200:
            return [_sanitize(x) for x in r.json()]
    except Exception:
        pass
    return None


def _sb_post(table: str, data) -> Optional[Any]:
    try:
        r = requests.post(f"{SUPABASE_URL.rstrip('/')}/rest/v1/{table}",
                          json=data, headers=_headers(), timeout=5)
        if r.status_code in (200, 201):
            j = r.json()
            return ([_sanitize(x) for x in j] if isinstance(j, list) else _sanitize(j))
    except Exception:
        pass
    return None


def _sb_patch(table: str, filter_str: str, data: dict) -> bool:
    try:
        r = requests.patch(
            f"{SUPABASE_URL.rstrip('/')}/rest/v1/{table}?{filter_str}",
            json=data, headers=_headers("return=minimal"), timeout=5
        )
        return r.status_code in (200, 204)
    except Exception:
        return False


def _sb_delete(table: str, filter_str: str) -> bool:
    try:
        r = requests.delete(
            f"{SUPABASE_URL.rstrip('/')}/rest/v1/{table}?{filter_str}",
            headers=_headers("return=minimal"), timeout=5
        )
        return r.status_code in (200, 204)
    except Exception:
        return False


# ── Leads CRUD ─────────────────────────────────────────────────────────────────

def get_all_leads(
    search: Optional[str] = None,
    status: Optional[str] = None,
    category: Optional[str] = None,
    pipeline_stage: Optional[str] = None,
    assigned_to: Optional[str] = None,
    industry: Optional[str] = None,
    lead_source: Optional[str] = None,
    sort_by: str = "score",
    sort_dir: str = "desc",
) -> List[Dict[str, Any]]:
    initialize_local_store()

    if is_supabase_configured():
        filters = ""
        if status and status != "All":
            filters += f"&status=eq.{status}"
        if category and category != "All":
            filters += f"&category=eq.{category.upper()}"
        if pipeline_stage and pipeline_stage != "All":
            filters += f"&pipeline_stage=eq.{pipeline_stage}"
        if assigned_to and assigned_to != "All":
            filters += f"&assigned_to=eq.{assigned_to}"
        if industry and industry != "All":
            filters += f"&industry=eq.{industry}"
        if lead_source and lead_source != "All":
            filters += f"&lead_source=eq.{lead_source}"

        results = _sb_get("leads", filters)
        if results is not None:
            if search:
                s = search.lower()
                results = [l for l in results if (
                    s in str(l.get("name", "")).lower() or
                    s in str(l.get("email", "")).lower() or
                    s in str(l.get("company", "")).lower()
                )]
            return _sort_leads(results, sort_by, sort_dir)

    # Local fallback
    leads = [_sanitize(l) for l in LOCAL_LEADS]
    if status and status != "All":
        leads = [l for l in leads if str(l.get("status", "")).lower() == status.lower()]
    if category and category != "All":
        leads = [l for l in leads if str(l.get("category", "")).upper() == category.upper()]
    if pipeline_stage and pipeline_stage != "All":
        leads = [l for l in leads if str(l.get("pipeline_stage", "")).lower() == pipeline_stage.lower()]
    if assigned_to and assigned_to != "All":
        leads = [l for l in leads if str(l.get("assigned_to", "")) == assigned_to]
    if industry and industry != "All":
        leads = [l for l in leads if str(l.get("industry", "")).lower() == industry.lower()]
    if lead_source and lead_source != "All":
        leads = [l for l in leads if str(l.get("lead_source", "")).lower() == lead_source.lower()]
    if search:
        s = search.lower()
        leads = [l for l in leads if (
            s in str(l.get("name", "")).lower() or
            s in str(l.get("email", "")).lower() or
            s in str(l.get("company", "")).lower()
        )]
    return _sort_leads(leads, sort_by, sort_dir)


def _sort_leads(leads: List[Dict], sort_by: str, sort_dir: str) -> List[Dict]:
    reverse = sort_dir.lower() != "asc"
    valid_fields = {"score", "conversion_probability", "created_at", "name", "company", "company_size"}
    field = sort_by if sort_by in valid_fields else "score"
    try:
        return sorted(leads, key=lambda x: (x.get(field) or 0), reverse=reverse)
    except Exception:
        return leads


def get_lead_by_id(lead_id: str) -> Optional[Dict[str, Any]]:
    initialize_local_store()
    if is_supabase_configured():
        results = _sb_get("leads", f"&id=eq.{lead_id}")
        if results:
            return results[0]
    for l in LOCAL_LEADS:
        if str(l.get("id")) == str(lead_id):
            return _sanitize(l)
    return None


def create_lead(lead_data: Dict[str, Any]) -> Dict[str, Any]:
    initialize_local_store()
    now = _now()
    lead_id = lead_data.get("id") or f"LEAD-{int(datetime.now().timestamp())}"
    record = _sanitize({
        "id": str(lead_id),
        "name": lead_data.get("name", "New Prospect"),
        "email": lead_data.get("email", ""),
        "phone": lead_data.get("phone", ""),
        "company": lead_data.get("company", ""),
        "job_title": lead_data.get("job_title", ""),
        "industry": lead_data.get("industry", "SaaS"),
        "company_size": _clean(lead_data.get("company_size"), 100.0),
        "location": lead_data.get("location", ""),
        "lead_source": lead_data.get("lead_source", "Website"),
        "campaign": lead_data.get("campaign", ""),
        "product_interest": lead_data.get("product_interest", ""),
        "budget_range": lead_data.get("budget_range", "Unknown"),
        "website_visits": _clean(lead_data.get("website_visits"), 0.0),
        "page_views": _clean(lead_data.get("page_views"), 0.0),
        "pricing_page_visits": _clean(lead_data.get("pricing_page_visits"), 0.0),
        "demo_requested": lead_data.get("demo_requested", "No"),
        "email_opens": _clean(lead_data.get("email_opens"), 0.0),
        "form_completions": _clean(lead_data.get("form_completions"), 0.0),
        "content_downloads": _clean(lead_data.get("content_downloads"), 0.0),
        "previous_interactions": _clean(lead_data.get("previous_interactions"), 0.0),
        "response_time_hours": _clean(lead_data.get("response_time_hours"), 0.0),
        "num_calls": _clean(lead_data.get("num_calls"), 0.0),
        "num_meetings": _clean(lead_data.get("num_meetings"), 0.0),
        "score": _clean(lead_data.get("score"), 0),
        "conversion_probability": _clean(lead_data.get("conversion_probability"), 0.0),
        "category": str(lead_data.get("category", "COLD")).upper(),
        "priority": lead_data.get("priority", "Low"),
        "status": lead_data.get("status", "New"),
        "pipeline_stage": lead_data.get("pipeline_stage", "New"),
        "assigned_to": lead_data.get("assigned_to"),
        "tags": lead_data.get("tags", []),
        "converted": lead_data.get("converted"),
        "outcome": lead_data.get("outcome"),
        "created_at": now,
        "updated_at": now,
    })

    if is_supabase_configured():
        result = _sb_post("leads", record)
        if result:
            r = result[0] if isinstance(result, list) else result
            _add_activity(lead_id, "lead_created", f"Lead '{record['name']}' created.")
            return r

    LOCAL_LEADS.insert(0, record)
    _add_activity(lead_id, "lead_created", f"Lead '{record['name']}' created.")
    return record


def update_lead(lead_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    initialize_local_store()
    updates["updated_at"] = _now()
    clean_updates = _sanitize(updates)

    if is_supabase_configured():
        ok = _sb_patch("leads", f"id=eq.{lead_id}", clean_updates)
        if ok:
            _log_update_activities(lead_id, updates)
            return get_lead_by_id(lead_id)

    for idx, l in enumerate(LOCAL_LEADS):
        if str(l.get("id")) == str(lead_id):
            LOCAL_LEADS[idx].update(clean_updates)
            _log_update_activities(lead_id, updates)
            return LOCAL_LEADS[idx]
    return None


def _log_update_activities(lead_id: str, updates: Dict):
    if "status" in updates:
        _add_activity(lead_id, "status_changed", f"Status changed to '{updates['status']}'.")
    if "pipeline_stage" in updates:
        _add_activity(lead_id, "pipeline_changed", f"Pipeline stage changed to '{updates['pipeline_stage']}'.")
    if "assigned_to" in updates and updates["assigned_to"]:
        _add_activity(lead_id, "assigned", f"Lead assigned to '{updates['assigned_to']}'.")
    if "score" in updates:
        _add_activity(lead_id, "score_calculated", f"Score updated to {updates['score']}.")


def delete_lead(lead_id: str) -> bool:
    initialize_local_store()
    if is_supabase_configured():
        if _sb_delete("leads", f"id=eq.{lead_id}"):
            return True

    global LOCAL_LEADS
    before = len(LOCAL_LEADS)
    LOCAL_LEADS = [l for l in LOCAL_LEADS if str(l.get("id")) != str(lead_id)]
    return len(LOCAL_LEADS) < before


# ── Notes ─────────────────────────────────────────────────────────────────────

def get_lead_notes(lead_id: str) -> List[Dict[str, Any]]:
    initialize_local_store()
    if is_supabase_configured():
        results = _sb_get("notes", f"&lead_id=eq.{lead_id}&order=created_at.desc")
        if results is not None:
            return results
    notes = [_sanitize(n) for n in LOCAL_NOTES if str(n.get("lead_id")) == str(lead_id)]
    return sorted(notes, key=lambda x: str(x.get("created_at", "")), reverse=True)


def add_lead_note(lead_id: str, note_text: str, author: str = "User") -> Dict[str, Any]:
    initialize_local_store()
    record = _sanitize({
        "id": str(uuid.uuid4()),
        "lead_id": lead_id,
        "note": note_text,
        "author": author,
        "created_at": _now(),
    })
    if is_supabase_configured():
        result = _sb_post("notes", record)
        if result:
            r = result[0] if isinstance(result, list) else result
            _add_activity(lead_id, "note_added", f"Note added: \"{note_text[:80]}\"")
            return r
    LOCAL_NOTES.insert(0, record)
    _add_activity(lead_id, "note_added", f"Note added: \"{note_text[:80]}\"")
    return record


# ── Activities ────────────────────────────────────────────────────────────────

def get_lead_activities(lead_id: str) -> List[Dict[str, Any]]:
    initialize_local_store()
    if is_supabase_configured():
        results = _sb_get("lead_activities", f"&lead_id=eq.{lead_id}&order=created_at.desc")
        if results is not None:
            return results
    acts = [_sanitize(a) for a in LOCAL_ACTIVITIES if str(a.get("lead_id")) == str(lead_id)]
    return sorted(acts, key=lambda x: str(x.get("created_at", "")), reverse=True)


def get_recent_activities(limit: int = 15) -> List[Dict[str, Any]]:
    initialize_local_store()
    if is_supabase_configured():
        results = _sb_get("lead_activities", f"&order=created_at.desc&limit={limit}")
        if results is not None:
            return results[:limit]
    acts = [_sanitize(a) for a in LOCAL_ACTIVITIES]
    return sorted(acts, key=lambda x: str(x.get("created_at", "")), reverse=True)[:limit]


# ── CSV Import ────────────────────────────────────────────────────────────────

def import_csv_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Import pre-validated, pre-scored records.
    Deduplicates by email or id.
    Returns summary dict.
    """
    initialize_local_store()
    imported = 0
    duplicates = 0
    failed = 0

    # Build existing id/email sets
    existing_ids = {str(l.get("id")) for l in LOCAL_LEADS}
    existing_emails = {str(l.get("email", "")).lower() for l in LOCAL_LEADS if l.get("email")}

    for rec in records:
        rec_id = str(rec.get("id", ""))
        rec_email = str(rec.get("email", "")).lower()
        if rec_id in existing_ids or (rec_email and rec_email in existing_emails):
            duplicates += 1
            continue
        try:
            created = create_lead(rec)
            imported += 1
            existing_ids.add(str(created.get("id")))
            if rec_email:
                existing_emails.add(rec_email)
        except Exception as e:
            print(f"[Import] Failed record: {e}")
            failed += 1

    return {"imported": imported, "duplicates": duplicates, "failed": failed}


# ── Dashboard & Analytics ─────────────────────────────────────────────────────

def get_dashboard_summary() -> Dict[str, Any]:
    leads = get_all_leads(sort_by="score", sort_dir="desc")
    total = len(leads)
    hot = sum(1 for l in leads if str(l.get("category", "")).upper() == "HOT")
    warm = sum(1 for l in leads if str(l.get("category", "")).upper() == "WARM")
    cold = sum(1 for l in leads if str(l.get("category", "")).upper() == "COLD")

    # Pipeline counts by stage
    pipeline_counts = {}
    for stage in PIPELINE_STAGES:
        pipeline_counts[stage] = sum(1 for l in leads if str(l.get("pipeline_stage", "")) == stage or str(l.get("status", "")) == stage)

    # Converted / Won / Lost
    won = sum(1 for l in leads if str(l.get("pipeline_stage", "")) == "Won" or l.get("converted") is True)
    lost = sum(1 for l in leads if str(l.get("pipeline_stage", "")) == "Lost" or l.get("converted") is False)
    resolved = won + lost
    conversion_rate = round((won / resolved * 100), 1) if resolved > 0 else 0.0

    # Top priority leads (by score, status not Won/Lost)
    priority_leads = [l for l in leads if str(l.get("pipeline_stage", "")) not in ("Won", "Lost")][:10]

    # Recent activity
    recent_acts = get_recent_activities(10)

    # Recent leads
    recent_leads = sorted(leads, key=lambda x: str(x.get("created_at", "")), reverse=True)[:10]

    return {
        "total_leads": total,
        "hot_leads": hot,
        "warm_leads": warm,
        "cold_leads": cold,
        "won_leads": won,
        "lost_leads": lost,
        "conversion_rate": conversion_rate,
        "pipeline_counts": pipeline_counts,
        "priority_leads": priority_leads,
        "recent_activities": recent_acts,
        "recent_leads": recent_leads,
    }


def get_analytics() -> Dict[str, Any]:
    leads = get_all_leads()
    total = len(leads)
    hot = sum(1 for l in leads if str(l.get("category", "")).upper() == "HOT")
    warm = sum(1 for l in leads if str(l.get("category", "")).upper() == "WARM")
    cold = sum(1 for l in leads if str(l.get("category", "")).upper() == "COLD")

    won = sum(1 for l in leads if str(l.get("pipeline_stage", "")) == "Won" or l.get("converted") is True)
    lost = sum(1 for l in leads if str(l.get("pipeline_stage", "")) == "Lost" or l.get("converted") is False)
    resolved = won + lost
    conversion_rate = round((won / resolved * 100), 1) if resolved > 0 else 0.0

    # Per-source breakdown
    from collections import defaultdict
    source_map: Dict[str, List] = defaultdict(list)
    for l in leads:
        src = l.get("lead_source") or "Unknown"
        source_map[src].append(l)

    source_stats = []
    for src, src_leads in sorted(source_map.items(), key=lambda x: -len(x[1])):
        src_won = sum(1 for l in src_leads if str(l.get("pipeline_stage", "")) == "Won" or l.get("converted") is True)
        src_lost = sum(1 for l in src_leads if str(l.get("pipeline_stage", "")) == "Lost" or l.get("converted") is False)
        src_res = src_won + src_lost
        src_rate = round(src_won / src_res * 100, 1) if src_res > 0 else 0.0
        avg_score = round(sum(l.get("score") or 0 for l in src_leads) / len(src_leads), 1)
        source_stats.append({
            "source": src,
            "total": len(src_leads),
            "won": src_won,
            "conversion_rate": src_rate,
            "avg_score": avg_score,
        })

    # Per-industry breakdown
    industry_map: Dict[str, List] = defaultdict(list)
    for l in leads:
        ind = l.get("industry") or "Unknown"
        industry_map[ind].append(l)

    industry_stats = []
    for ind, ind_leads in sorted(industry_map.items(), key=lambda x: -len(x[1])):
        ind_won = sum(1 for l in ind_leads if str(l.get("pipeline_stage", "")) == "Won" or l.get("converted") is True)
        ind_res_total = sum(1 for l in ind_leads if l.get("converted") is not None)
        ind_rate = round(ind_won / ind_res_total * 100, 1) if ind_res_total > 0 else 0.0
        avg_score = round(sum(l.get("score") or 0 for l in ind_leads) / len(ind_leads), 1)
        industry_stats.append({
            "industry": ind,
            "total": len(ind_leads),
            "won": ind_won,
            "conversion_rate": ind_rate,
            "avg_score": avg_score,
        })

    # Score distribution
    scores = [l.get("score") or 0 for l in leads]
    score_dist = [
        {"range": "0-19", "count": sum(1 for s in scores if s < 20)},
        {"range": "20-39", "count": sum(1 for s in scores if 20 <= s < 40)},
        {"range": "40-59", "count": sum(1 for s in scores if 40 <= s < 60)},
        {"range": "60-79", "count": sum(1 for s in scores if 60 <= s < 80)},
        {"range": "80-100", "count": sum(1 for s in scores if s >= 80)},
    ]

    return {
        "overview": {
            "total_leads": total,
            "hot_leads": hot,
            "warm_leads": warm,
            "cold_leads": cold,
            "won_leads": won,
            "lost_leads": lost,
            "conversion_rate": conversion_rate,
        },
        "source_breakdown": source_stats,
        "industry_breakdown": industry_stats,
        "score_distribution": score_dist,
    }
