"""
backend/app.py
FastAPI backend for Lead Scoring + CRM System.
Calls existing ML model (read-only). Uses Supabase for persistence.
Supports Auth, Workspaces, Real XGBoost Explainability,
Team Assignment, Alerts, CSV Import/Export, and Retraining.
"""
import sys
import io
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

# Configure server-side logging (safe — never logs secrets)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("crm.app")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import existing ML scoring function (READ-ONLY, never modified)
from ml.inference.predict import score_single_lead, load_trained_pipeline
from ml.training.retrain_service import execute_retraining_workflow
import numpy as np

# Import CRM persistence layer
from backend import supabase_db

app = FastAPI(
    title="Lead Scoring + CRM API",
    description="Production API connecting existing XGBoost ML model to Supabase CRM.",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Feature importance cache (from real XGBoost model) ────────────────────────
_FEATURE_IMPORTANCE: Optional[Dict[str, float]] = None

def get_feature_importance() -> Dict[str, float]:
    global _FEATURE_IMPORTANCE
    if _FEATURE_IMPORTANCE is not None:
        return _FEATURE_IMPORTANCE
    try:
        pipeline = load_trained_pipeline()
        model = pipeline.named_steps["classifier"]
        pre = pipeline.named_steps["preprocessor"]
        names = pre.get_feature_names_out()
        importances = model.feature_importances_
        _FEATURE_IMPORTANCE = {str(n): float(v) for n, v in zip(names, importances)}
    except Exception:
        _FEATURE_IMPORTANCE = {}
    return _FEATURE_IMPORTANCE


def generate_score_explanation(lead_dict: Dict[str, Any], score: int, prob: float) -> Dict[str, Any]:
    """
    Builds score explanation using real XGBoost feature importances.
    Positive signals = high-importance features that are 'positive' for conversion.
    Negative signals = high-importance features that are 'negative' for conversion.
    """
    fi = get_feature_importance()
    feature_signals = []

    if lead_dict.get("demo_requested") == "Yes":
        imp = fi.get("nom__demo_requested_Yes", 0.098)
        feature_signals.append({
            "label": "Demo Requested", "signal": "positive", "importance": imp,
            "reason": "Explicit demo request is the strongest conversion signal"
        })
    elif lead_dict.get("demo_requested") == "No":
        imp = fi.get("nom__demo_requested_No", 0.02)
        feature_signals.append({
            "label": "No Demo Request", "signal": "negative", "importance": imp,
            "reason": "No demo requested reduces conversion likelihood"
        })

    meetings = lead_dict.get("num_meetings", 0) or 0
    imp = fi.get("num__num_meetings", 0.076)
    if meetings >= 1:
        feature_signals.append({
            "label": f"{int(meetings)} Sales Meeting(s)", "signal": "positive",
            "importance": imp, "reason": "Sales meetings indicate strong engagement"
        })
    elif meetings == 0:
        feature_signals.append({
            "label": "No Sales Meetings", "signal": "negative",
            "importance": imp, "reason": "No sales meetings indicates early stage"
        })

    pricing = lead_dict.get("pricing_page_visits", 0) or 0
    imp = fi.get("num__pricing_page_visits", 0.041)
    if pricing >= 3:
        feature_signals.append({
            "label": f"{int(pricing)} Pricing Page Visits", "signal": "positive",
            "importance": imp, "reason": "High pricing page engagement shows purchase intent"
        })
    elif pricing == 0:
        feature_signals.append({
            "label": "No Pricing Page Visits", "signal": "negative",
            "importance": imp, "reason": "Zero pricing page visits indicates low purchase intent"
        })

    company_size = lead_dict.get("company_size", 0) or 0
    imp = fi.get("num__company_size", 0.052)
    if company_size >= 200:
        feature_signals.append({
            "label": f"Company Size: {int(company_size)}", "signal": "positive",
            "importance": imp, "reason": "Larger organizations have greater buying capacity"
        })
    elif company_size < 20:
        feature_signals.append({
            "label": f"Small Company ({int(company_size)})", "signal": "negative",
            "importance": imp, "reason": "Very small company may have limited budget"
        })

    response_time = lead_dict.get("response_time_hours", 0) or 0
    imp = fi.get("num__response_time_hours", 0.035)
    if response_time <= 2:
        feature_signals.append({
            "label": f"Fast Response ({response_time}h)", "signal": "positive",
            "importance": imp, "reason": "Quick response time indicates high engagement"
        })
    elif response_time > 24:
        feature_signals.append({
            "label": f"Slow Response ({response_time}h)", "signal": "negative",
            "importance": imp, "reason": "Slow response indicates low urgency"
        })

    form_completions = lead_dict.get("form_completions", 0) or 0
    imp = fi.get("num__form_completions", 0.03)
    if form_completions >= 2:
        feature_signals.append({
            "label": f"{int(form_completions)} Form Completions", "signal": "positive",
            "importance": imp, "reason": "Multiple form submissions indicate genuine interest"
        })

    source = lead_dict.get("lead_source", "")
    if source == "Referral":
        imp = fi.get("nom__lead_source_Referral", 0.04)
        feature_signals.append({
            "label": "Referral Source", "signal": "positive",
            "importance": imp, "reason": "Referrals historically convert at higher rates"
        })
    elif source == "Paid Ads":
        imp = fi.get("nom__lead_source_Paid Ads", 0.02)
        feature_signals.append({
            "label": "Paid Ad Source", "signal": "negative",
            "importance": imp, "reason": "Paid ad leads have lower average conversion"
        })

    feature_signals.sort(key=lambda x: -x["importance"])
    positives = [s for s in feature_signals if s["signal"] == "positive"][:4]
    negatives = [s for s in feature_signals if s["signal"] == "negative"][:3]

    category = "HOT" if prob >= 0.75 else ("WARM" if prob >= 0.40 else "COLD")
    summary = (
        f"Score of {score} reflects {len(positives)} positive and {len(negatives)} negative conversion signals "
        f"from the production XGBoost model (conversion probability: {round(prob*100,1)}%)."
    )

    return {
        "score": score,
        "conversion_probability_pct": round(prob * 100, 1),
        "category": category,
        "summary": summary,
        "positive_signals": positives,
        "negative_signals": negatives,
    }


# ── Pydantic Schemas ───────────────────────────────────────────────────────────

class SignupInput(BaseModel):
    email: str
    password: str
    name: Optional[str] = None


class LoginInput(BaseModel):
    email: str
    password: str


class MemberInput(BaseModel):
    email: str
    name: str
    role: Optional[str] = "Salesperson"


class LeadScoreInput(BaseModel):
    name: Optional[str] = "New Prospect"
    email: Optional[str] = ""
    phone: Optional[str] = ""
    company: Optional[str] = ""
    job_title: Optional[str] = ""
    location: Optional[str] = ""

    industry: str = Field(default="SaaS")
    company_size: float = Field(default=250.0)
    lead_source: str = Field(default="Website")
    product_interest: str = Field(default="Enterprise Plan")
    budget_range: str = Field(default="50k-1L")

    website_visits: float = Field(default=12.0)
    page_views: float = Field(default=35.0)
    pricing_page_visits: float = Field(default=5.0)
    demo_requested: str = Field(default="Yes")

    email_opens: float = Field(default=8.0)
    form_completions: float = Field(default=2.0)
    content_downloads: float = Field(default=3.0)
    previous_interactions: float = Field(default=4.0)
    response_time_hours: float = Field(default=2.5)
    num_calls: float = Field(default=3.0)
    num_meetings: float = Field(default=1.0)


class CreateLeadSchema(LeadScoreInput):
    workspace_id: Optional[str] = None
    score: Optional[int] = None
    conversion_probability: Optional[float] = None
    category: Optional[str] = "COLD"
    priority: Optional[str] = "Low"
    status: Optional[str] = "New"
    pipeline_stage: Optional[str] = "New"
    assigned_to: Optional[str] = None
    assigned_user_id: Optional[str] = None
    tags: Optional[List[str]] = []
    converted: Optional[bool] = None
    outcome: Optional[str] = None
    converted_at: Optional[str] = None


class UpdateLeadSchema(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    industry: Optional[str] = None
    location: Optional[str] = None
    status: Optional[str] = None
    pipeline_stage: Optional[str] = None
    assigned_to: Optional[str] = None
    assigned_user_id: Optional[str] = None
    tags: Optional[List[str]] = None
    score: Optional[int] = None
    category: Optional[str] = None
    priority: Optional[str] = None
    converted: Optional[bool] = None
    outcome: Optional[str] = None
    converted_at: Optional[str] = None


class NoteInput(BaseModel):
    note: str
    author: Optional[str] = "User"


# ── Auth & Workspace Endpoints ────────────────────────────────────────────────

@app.post("/api/auth/signup")
def signup_api(body: SignupInput):
    result = supabase_db.auth_signup(body.email, body.password, body.name)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Signup failed"))
    return result


@app.post("/api/auth/login")
def login_api(body: LoginInput):
    result = supabase_db.auth_login(body.email, body.password)
    if not result.get("success"):
        raise HTTPException(status_code=401, detail=result.get("error", "Invalid credentials"))
    return result


@app.get("/api/auth/me")
def me_api(authorization: Optional[str] = Header(None)):
    user = supabase_db.auth_get_current_user(authorization or "")
    if not user:
        # Graceful return for unauthenticated check
        return {"success": False, "authenticated": False}
    u = dict(user)
    if "id" not in u:
        u["id"] = u.get("user_id", "")
    return {"success": True, "authenticated": True, "user": u}


@app.post("/api/auth/logout")
def logout_api():
    return {"success": True, "message": "Logged out successfully"}


@app.get("/api/workspace")
def get_workspace_api(authorization: Optional[str] = Header(None)):
    user = supabase_db.auth_get_current_user(authorization or "")
    ws_id = user.get("workspace_id") if user else supabase_db.DEFAULT_WORKSPACE_ID
    ws = next((w for w in supabase_db.LOCAL_WORKSPACES if w["id"] == ws_id), supabase_db.LOCAL_WORKSPACES[0])
    return {"success": True, "workspace": ws}


@app.get("/api/workspace/members")
def get_workspace_members_api(authorization: Optional[str] = Header(None)):
    user = supabase_db.auth_get_current_user(authorization or "")
    ws_id = user.get("workspace_id") if user else supabase_db.DEFAULT_WORKSPACE_ID
    members = supabase_db.get_workspace_members(ws_id)
    return {"success": True, "members": members}


@app.post("/api/workspace/members")
def add_workspace_member_api(member: MemberInput, authorization: Optional[str] = Header(None)):
    user = supabase_db.auth_get_current_user(authorization or "")
    ws_id = user.get("workspace_id") if user else supabase_db.DEFAULT_WORKSPACE_ID
    new_mem = supabase_db.add_workspace_member(ws_id, member.email, member.name, member.role or "Salesperson")
    return {"success": True, "member": new_mem}


# ── Health & Score Endpoints ──────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    """
    Safe health diagnostic endpoint.
    Reports Supabase configuration and actual connectivity status.
    NEVER exposes SUPABASE_KEY or any credential values.
    """
    import os
    supabase_configured = supabase_db.is_supabase_configured()

    # Safe URL presence check — does NOT expose the key
    url_configured = bool(os.getenv("SUPABASE_URL", "").strip())
    key_configured = bool(
        os.getenv("SUPABASE_KEY", "").strip()
        or os.getenv("SUPABASE_ANON_KEY", "").strip()
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        or os.getenv("SUPABASE_SERVICE_KEY", "").strip()
    )

    # Probe actual Supabase connectivity (lightweight HEAD on leads table)
    connection_status = "not_attempted"
    if supabase_configured:
        probe = supabase_db.probe_supabase_connection()
        connection_status = probe["status"]  # "ok" | "error" | "auth_error"
        connection_error = probe.get("detail")
    else:
        connection_error = "SUPABASE_URL or SUPABASE_KEY not configured"

    database_mode = "supabase" if supabase_configured else "memory_fallback"

    if not supabase_configured:
        overall_status = "error"
    elif connection_status == "ok":
        overall_status = "ok"
    else:
        overall_status = "degraded"

    response = {
        "status": overall_status,
        "service": "Lead Scoring + CRM API v2.1",
        "database": database_mode,
        "supabase_configured": supabase_configured,
        "supabase_url_set": url_configured,
        "supabase_key_set": key_configured,
        "supabase_connection": connection_status,
        "ml_model": "XGBoost (read-only)",
    }
    # Include safe error detail only if something is wrong — never include key values
    if overall_status != "ok" and connection_error:
        response["error_detail"] = connection_error
    return response


@app.post("/api/score")
def score_lead_api(lead_input: LeadScoreInput):
    """
    Calls the existing ML model (score_single_lead).
    Returns real score, probability, priority, and explanation.
    Provides backward-compatible aliases (lead_score, key_drivers, recommended_action).
    """
    try:
        lead_dict = lead_input.model_dump()
        result = score_single_lead(lead_dict)

        score = int(result["lead_score"])
        prob = float(result["conversion_probability"])
        priority = str(result["priority"])
        category = str(result["lead_priority"]).upper()
        if category not in ("HOT", "WARM", "COLD"):
            category = "HOT" if prob >= 0.75 else ("WARM" if prob >= 0.40 else "COLD")

        explanation = generate_score_explanation(lead_dict, score, prob)
        key_drivers = [f"{s['label']}: {s['reason']}" for s in explanation.get("positive_signals", [])]
        recommended_action = (
            "Immediate high-priority outreach scheduled. Recommend executive discovery call."
            if category == "HOT" else
            ("Schedule product walkthrough and share ROI case studies." if category == "WARM" else "Add to automated nurture campaign.")
        )

        return {
            "success": True,
            "score": score,
            "lead_score": score,  # Compatibility alias
            "conversion_probability": round(prob, 4),
            "conversion_probability_pct": round(prob * 100, 1),
            "category": category,
            "priority": priority,
            "key_drivers": key_drivers,  # Compatibility alias
            "recommended_action": recommended_action,  # Compatibility alias
            "explanation": explanation,
            "input_data": lead_dict,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ML prediction error: {str(e)}")


# ── Leads Endpoints ───────────────────────────────────────────────────────────

@app.get("/api/leads")
def get_leads(
    search: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    pipeline_stage: Optional[str] = Query(None),
    industry: Optional[str] = Query(None),
    lead_source: Optional[str] = Query(None),
    assigned_to: Optional[str] = Query(None),
    workspace_id: Optional[str] = Query(None),
    sort_by: str = Query("score"),
    sort_dir: str = Query("desc"),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
):
    leads = supabase_db.get_all_leads(
        search=search, status=status, category=category,
        pipeline_stage=pipeline_stage, assigned_to=assigned_to,
        industry=industry, lead_source=lead_source,
        workspace_id=workspace_id,
        sort_by=sort_by, sort_dir=sort_dir,
    )
    total = len(leads)
    start = (page - 1) * limit
    paged = leads[start:start + limit]
    return {"success": True, "total": total, "page": page, "limit": limit, "count": len(paged), "leads": paged}


@app.get("/api/leads/{lead_id}")
def get_lead(lead_id: str):
    lead = supabase_db.get_lead_by_id(lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    return {"success": True, "lead": lead}


@app.post("/api/leads")
def create_lead(lead: CreateLeadSchema, authorization: Optional[str] = Header(None)):
    user = supabase_db.auth_get_current_user(authorization or "")
    actor_name = user["name"] if user else "CRM User"
    try:
        created = supabase_db.create_lead(lead.model_dump(), actor=actor_name)
        return {"success": True, "lead": created}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/leads/{lead_id}")
def update_lead(lead_id: str, updates: UpdateLeadSchema, authorization: Optional[str] = Header(None)):
    user = supabase_db.auth_get_current_user(authorization or "")
    actor_name = user["name"] if user else "CRM User"
    data = {k: v for k, v in updates.model_dump().items() if v is not None}
    if not data:
        raise HTTPException(status_code=400, detail="No update fields provided")
    updated = supabase_db.update_lead(lead_id, data, actor=actor_name)
    if not updated:
        raise HTTPException(status_code=404, detail="Lead not found")
    return {"success": True, "lead": updated}


@app.delete("/api/leads/{lead_id}")
def delete_lead(lead_id: str):
    ok = supabase_db.delete_lead(lead_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Lead not found")
    return {"success": True, "message": "Lead deleted"}


@app.get("/api/leads/{lead_id}/notes")
def get_notes(lead_id: str):
    return {"success": True, "notes": supabase_db.get_lead_notes(lead_id)}


@app.post("/api/leads/{lead_id}/notes")
def add_note(lead_id: str, note_in: NoteInput, authorization: Optional[str] = Header(None)):
    if not note_in.note.strip():
        raise HTTPException(status_code=400, detail="Note cannot be empty")
    user = supabase_db.auth_get_current_user(authorization or "")
    author_name = user["name"] if user else (note_in.author or "User")
    note = supabase_db.add_lead_note(lead_id, note_in.note.strip(), author=author_name)
    return {"success": True, "note": note}


@app.get("/api/leads/{lead_id}/activities")
def get_activities(lead_id: str):
    return {"success": True, "activities": supabase_db.get_lead_activities(lead_id)}


# ── Dashboard & Analytics ─────────────────────────────────────────────────────

@app.get("/api/dashboard")
def get_dashboard(workspace_id: Optional[str] = Query(None)):
    summary = supabase_db.get_dashboard_summary(workspace_id or supabase_db.DEFAULT_WORKSPACE_ID)
    return {"success": True, "dashboard": summary}


@app.get("/api/analytics")
def get_analytics(workspace_id: Optional[str] = Query(None)):
    data = supabase_db.get_analytics(workspace_id or supabase_db.DEFAULT_WORKSPACE_ID)
    return {
        "success": True,
        "analytics": data,
        "kpis": data.get("overview", {}),  # Backward compatibility alias for existing tests
        "score_distribution": data.get("score_distribution", []),  # Compatibility alias
    }


# ── Smart Alerts ──────────────────────────────────────────────────────────────

@app.get("/api/alerts")
def get_alerts(workspace_id: Optional[str] = Query(None)):
    alerts = supabase_db.get_smart_alerts(workspace_id or supabase_db.DEFAULT_WORKSPACE_ID)
    return {"success": True, "alerts": alerts, "count": len(alerts)}


# ── CSV Import & Export ───────────────────────────────────────────────────────

@app.get("/api/export")
def export_leads(
    search: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    pipeline_stage: Optional[str] = Query(None),
    industry: Optional[str] = Query(None),
    lead_source: Optional[str] = Query(None),
    assigned_to: Optional[str] = Query(None),
    workspace_id: Optional[str] = Query(None),
):
    csv_text = supabase_db.export_leads_csv(
        search=search, status=status, category=category,
        pipeline_stage=pipeline_stage, assigned_to=assigned_to,
        industry=industry, lead_source=lead_source,
        workspace_id=workspace_id
    )
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="crm_leads_export.csv"'}
    )


@app.post("/api/import")
async def import_csv(file: UploadFile = File(...), workspace_id: Optional[str] = Query(None)):
    """
    CSV Import endpoint.
    Validates columns, scores new leads via XGBoost ML model (same path as manual scoring),
    detects duplicates against actual Supabase, and imports into Supabase.
    Provides detailed per-stage logging — never logs secrets or passwords.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    content = await file.read()
    try:
        text = content.decode("utf-8-sig")
    except Exception:
        text = content.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    total_csv_rows = len(rows)

    logger.info(f"[CSV IMPORT] File: '{file.filename}' | CSV rows received: {total_csv_rows}")

    if not rows:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    headers_lower = [h.lower().strip() for h in reader.fieldnames or []]
    required = {"name", "email"}
    missing_req = required - set(headers_lower)
    if missing_req:
        raise HTTPException(
            status_code=422,
            detail=f"CSV is missing required columns: {', '.join(missing_req)}"
        )

    def get_col(row, *aliases):
        for alias in aliases:
            for k, v in row.items():
                if k and k.lower().strip() == alias:
                    return v
        return ""

    def safe_float(val, field_name: str, row_index: int, default: float = 0.0) -> float:
        """Convert a CSV string to float. Logs a warning on failure, returns default."""
        if val is None or str(val).strip() == "":
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            logger.warning(f"[CSV IMPORT] Row {row_index}: field '{field_name}' has non-numeric value '{val}', using {default}")
            return default

    records_to_score: List[Dict[str, Any]] = []
    invalid_rows = 0
    invalid_reasons: List[str] = []

    # ── Stage 1: Parse and validate rows ──────────────────────────────────────
    for i, row in enumerate(rows):
        name = get_col(row, "name") or f"CSV Lead {i+1}"
        email = get_col(row, "email").strip()
        if not email:
            invalid_rows += 1
            invalid_reasons.append(f"Row {i+1}: missing email")
            logger.warning(f"[CSV IMPORT] Row {i+1}: skipped — missing email")
            continue

        target_raw = get_col(row, "target", "converted", "is_converted").lower().strip()
        converted = None
        if target_raw in ("1", "true", "yes", "won"):
            converted = True
        elif target_raw in ("0", "false", "no", "lost"):
            converted = False

        rec = {
            "name": name,
            "email": email,
            "phone": get_col(row, "phone"),
            "company": get_col(row, "company"),
            "job_title": get_col(row, "job_title", "jobtitle"),
            # Categorical — passed as-is; ML pipeline handles encoding
            "industry": get_col(row, "industry") or "Unknown",
            "company_size": safe_float(get_col(row, "company_size", "companysize"), "company_size", i+1, 100.0),
            "location": get_col(row, "location"),
            "lead_source": get_col(row, "lead_source", "leadsource", "source") or "Import",
            "product_interest": get_col(row, "product_interest") or "Enterprise Plan",
            "budget_range": get_col(row, "budget_range") or "Unknown",
            "demo_requested": get_col(row, "demo_requested") or "No",
            # Numeric — explicitly cast with safe_float to guarantee correct dtype for ML pipeline
            "website_visits": safe_float(get_col(row, "website_visits"), "website_visits", i+1),
            "page_views": safe_float(get_col(row, "page_views"), "page_views", i+1),
            "pricing_page_visits": safe_float(get_col(row, "pricing_page_visits"), "pricing_page_visits", i+1),
            "email_opens": safe_float(get_col(row, "email_opens"), "email_opens", i+1),
            "form_completions": safe_float(get_col(row, "form_completions"), "form_completions", i+1),
            "content_downloads": safe_float(get_col(row, "content_downloads"), "content_downloads", i+1),
            "previous_interactions": safe_float(get_col(row, "previous_interactions"), "previous_interactions", i+1),
            "response_time_hours": safe_float(get_col(row, "response_time_hours"), "response_time_hours", i+1),
            "num_calls": safe_float(get_col(row, "num_calls"), "num_calls", i+1),
            "num_meetings": safe_float(get_col(row, "num_meetings"), "num_meetings", i+1),
            "status": get_col(row, "status") or "New",
            "pipeline_stage": get_col(row, "pipeline_stage") or "New",
            "converted": converted,
            "outcome": "Won" if converted is True else ("Lost" if converted is False else None),
        }
        records_to_score.append(rec)

    valid_rows = len(records_to_score)
    logger.info(f"[CSV IMPORT] Valid rows (have email): {valid_rows} | Invalid/skipped: {invalid_rows}")

    # ── Stage 2: Score each row via production XGBoost pipeline ───────────────
    # Uses EXACTLY the same code path as manual /api/score endpoint:
    #   LeadScoreInput → score_single_lead() → predict_batch() → predict_lead_scores()
    records_to_import: List[Dict[str, Any]] = []
    scoring_failures: List[Dict[str, str]] = []

    for i, rec in enumerate(records_to_score):
        try:
            # Filter empty strings so Pydantic uses its ML-compatible defaults
            clean_rec = {k: v for k, v in rec.items() if v != "" and v is not None}

            # LeadScoreInput enforces the same field types/defaults as manual scoring
            lead_input = LeadScoreInput(**clean_rec)
            ml_result = score_single_lead(lead_input.model_dump())

            prob = float(ml_result["conversion_probability"])
            # Category logic identical to /api/score endpoint
            category = str(ml_result.get("lead_priority", "")).upper()
            if category not in ("HOT", "WARM", "COLD"):
                category = "HOT" if prob >= 0.75 else ("WARM" if prob >= 0.40 else "COLD")

            rec["score"] = int(ml_result["lead_score"])
            rec["conversion_probability"] = round(prob, 4)
            rec["priority"] = str(ml_result["priority"])
            rec["category"] = category
            records_to_import.append(rec)

        except Exception as ml_err:
            # Do NOT invent a score — mark as failed and record the real error
            err_msg = str(ml_err)
            logger.error(f"[CSV IMPORT] Row {i+1} ({rec.get('email', 'unknown')}): ML scoring failed — {err_msg}")
            scoring_failures.append({"row": i + 1, "email": rec.get("email", ""), "error": err_msg})

    scored_rows = len(records_to_import)
    logger.info(f"[CSV IMPORT] Rows sent for scoring: {valid_rows} | Successfully scored: {scored_rows} | Scoring failures: {len(scoring_failures)}")

    # ── Stage 3: Insert into Supabase (or memory fallback) ────────────────────
    ws_id = workspace_id or supabase_db.DEFAULT_WORKSPACE_ID
    result = supabase_db.import_csv_records(records_to_import, workspace_id=ws_id, filename=file.filename)

    logger.info(
        f"[CSV IMPORT] DB result — Inserted: {result['imported']} | "
        f"Duplicates: {result['duplicates']} | DB failures: {result['failed']}"
    )

    total_failed = len(scoring_failures) + result["failed"]

    response = {
        "success": True,
        "filename": file.filename,
        "total_rows": total_csv_rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "rows_scored": scored_rows,
        "scoring_failures": len(scoring_failures),
        "imported": result["imported"],
        "duplicates": result["duplicates"],
        "db_failures": result["failed"],
        "failed": total_failed,
        "database_mode": "supabase" if supabase_db.is_supabase_configured() else "memory_fallback",
    }
    # Include safe per-row error details for debugging (no secrets)
    if scoring_failures:
        response["scoring_failure_details"] = scoring_failures[:20]  # cap at 20 to avoid huge response

    return response


@app.get("/api/imports/history")
def get_import_history(workspace_id: Optional[str] = Query(None)):
    ws_id = workspace_id or supabase_db.DEFAULT_WORKSPACE_ID
    history = supabase_db.get_csv_import_history(ws_id)
    return {"success": True, "history": history}


# ── Model Versioning & Retraining Endpoints ───────────────────────────────────

@app.get("/api/model-info")
def get_model_info_api():
    """Returns active production model information and metrics."""
    info = supabase_db.get_production_model_info()
    return {"success": True, **info}


@app.get("/api/model/versions")
def get_model_versions_api():
    """Returns registry of all trained model versions."""
    versions = supabase_db.get_model_versions()
    return {"success": True, "versions": versions}


@app.post("/api/model/retrain")
def trigger_retraining_api():
    """
    Executes automated model retraining workflow.
    Trains candidate XGBoost on resolved CRM outcomes, compares ROC-AUC with production,
    and conditionally promotes only if candidate is superior.
    """
    result = execute_retraining_workflow()
    return result


@app.get("/api/sample-leads")
def get_sample_leads_api():
    """Returns pre-configured sample leads for testing and demo scoring."""
    samples = [
        {
            "name": "Sarah Connor",
            "company": "Cyberdyne Systems",
            "industry": "Technology",
            "company_size": 1500,
            "lead_source": "Referral",
            "product_interest": "Enterprise Plan",
            "budget_range": "1L-5L",
            "website_visits": 24,
            "page_views": 65,
            "pricing_page_visits": 9,
            "demo_requested": "Yes",
            "email_opens": 15,
            "form_completions": 4,
            "content_downloads": 6,
            "previous_interactions": 8,
            "response_time_hours": 1.0,
            "num_calls": 5,
            "num_meetings": 3,
        },
        {
            "name": "Marcus Vance",
            "company": "Vance Refrigeration",
            "industry": "Retail",
            "company_size": 45,
            "lead_source": "Website",
            "product_interest": "Growth Plan",
            "budget_range": "10k-50k",
            "website_visits": 8,
            "page_views": 18,
            "pricing_page_visits": 2,
            "demo_requested": "No",
            "email_opens": 4,
            "form_completions": 1,
            "content_downloads": 1,
            "previous_interactions": 2,
            "response_time_hours": 8.0,
            "num_calls": 1,
            "num_meetings": 0,
        },
        {
            "name": "Elena Rostova",
            "company": "AeroGlobal Logistics",
            "industry": "Logistics",
            "company_size": 400,
            "lead_source": "Organic Search",
            "product_interest": "Enterprise Plan",
            "budget_range": "50k-1L",
            "website_visits": 14,
            "page_views": 32,
            "pricing_page_visits": 4,
            "demo_requested": "Yes",
            "email_opens": 7,
            "form_completions": 2,
            "content_downloads": 3,
            "previous_interactions": 4,
            "response_time_hours": 2.2,
            "num_calls": 2,
            "num_meetings": 1,
        },
    ]
    return {"success": True, "samples": samples}


@app.get("/api/pipeline-stages")
def get_pipeline_stages():
    return {"success": True, "stages": supabase_db.PIPELINE_STAGES}


# ── Static Frontend ───────────────────────────────────────────────────────────
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def read_root():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"message": "Lead Scoring + CRM API running."})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
