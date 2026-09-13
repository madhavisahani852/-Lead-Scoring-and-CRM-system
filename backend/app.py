"""
backend/app.py
FastAPI backend for Lead Scoring + CRM System.
Calls existing ML model (read-only). Uses Supabase for persistence.
"""
import sys
import io
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import existing ML scoring function (READ-ONLY, never modified)
from ml.inference.predict import score_single_lead, load_trained_pipeline
import numpy as np

# Import CRM persistence layer
from backend import supabase_db

app = FastAPI(
    title="Lead Scoring + CRM API",
    description="Production API connecting existing XGBoost ML model to Supabase CRM.",
    version="2.0.0",
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

    # Map raw lead features to their engineered/encoded names
    feature_signals = []

    if lead_dict.get("demo_requested") == "Yes":
        imp = fi.get("nom__demo_requested_Yes", 0)
        feature_signals.append({"label": "Demo Requested", "signal": "positive", "importance": imp,
                                 "reason": "Explicit demo request is the strongest conversion signal"})
    elif lead_dict.get("demo_requested") == "No":
        imp = fi.get("nom__demo_requested_No", 0)
        feature_signals.append({"label": "No Demo Request", "signal": "negative", "importance": imp,
                                 "reason": "No demo requested reduces conversion likelihood"})

    meetings = lead_dict.get("num_meetings", 0) or 0
    imp = fi.get("num__num_meetings", 0)
    if meetings >= 1:
        feature_signals.append({"label": f"{int(meetings)} Sales Meeting(s)", "signal": "positive",
                                 "importance": imp, "reason": "Sales meetings indicate strong engagement"})
    elif meetings == 0:
        feature_signals.append({"label": "No Sales Meetings", "signal": "negative",
                                 "importance": imp, "reason": "No sales meetings indicates early stage"})

    pricing = lead_dict.get("pricing_page_visits", 0) or 0
    imp = fi.get("num__pricing_page_visits", 0)
    if pricing >= 3:
        feature_signals.append({"label": f"{int(pricing)} Pricing Page Visits", "signal": "positive",
                                 "importance": imp, "reason": "High pricing page engagement shows purchase intent"})
    elif pricing == 0:
        feature_signals.append({"label": "No Pricing Page Visits", "signal": "negative",
                                 "importance": imp, "reason": "Zero pricing page visits indicates low purchase intent"})

    company_size = lead_dict.get("company_size", 0) or 0
    imp = fi.get("num__company_size", 0)
    if company_size >= 200:
        feature_signals.append({"label": f"Company Size: {int(company_size)}", "signal": "positive",
                                 "importance": imp, "reason": "Larger organizations have greater buying capacity"})
    elif company_size < 20:
        feature_signals.append({"label": f"Small Company ({int(company_size)})", "signal": "negative",
                                 "importance": imp, "reason": "Very small company may have limited budget"})

    response_time = lead_dict.get("response_time_hours", 0) or 0
    imp = fi.get("num__response_time_hours", 0)
    if response_time <= 2:
        feature_signals.append({"label": f"Fast Response ({response_time}h)", "signal": "positive",
                                 "importance": imp, "reason": "Quick response time indicates high engagement"})
    elif response_time > 24:
        feature_signals.append({"label": f"Slow Response ({response_time}h)", "signal": "negative",
                                 "importance": imp, "reason": "Slow response indicates low urgency"})

    form_completions = lead_dict.get("form_completions", 0) or 0
    imp = fi.get("num__form_completions", 0)
    if form_completions >= 2:
        feature_signals.append({"label": f"{int(form_completions)} Form Completions", "signal": "positive",
                                 "importance": imp, "reason": "Multiple form submissions indicate genuine interest"})

    source = lead_dict.get("lead_source", "")
    if source == "Referral":
        imp = fi.get("nom__lead_source_Referral", 0)
        feature_signals.append({"label": "Referral Source", "signal": "positive",
                                 "importance": imp, "reason": "Referrals historically convert at higher rates"})
    elif source == "Paid Ads":
        imp = fi.get("nom__lead_source_Paid Ads", 0)
        feature_signals.append({"label": "Paid Ad Source", "signal": "negative",
                                 "importance": imp, "reason": "Paid ad leads have lower average conversion"})

    # Sort by importance
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
    score: Optional[int] = None
    conversion_probability: Optional[float] = None
    category: Optional[str] = "COLD"
    priority: Optional[str] = "Low"
    status: Optional[str] = "New"
    pipeline_stage: Optional[str] = "New"
    assigned_to: Optional[str] = None
    tags: Optional[List[str]] = []
    converted: Optional[bool] = None
    outcome: Optional[str] = None


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
    tags: Optional[List[str]] = None
    score: Optional[int] = None
    category: Optional[str] = None
    priority: Optional[str] = None
    converted: Optional[bool] = None
    outcome: Optional[str] = None


class NoteInput(BaseModel):
    note: str
    author: Optional[str] = "User"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Lead Scoring + CRM API v2.0",
        "supabase": supabase_db.is_supabase_configured(),
        "ml_model": "XGBoost (read-only)",
    }


@app.post("/api/score")
def score_lead_api(lead_input: LeadScoreInput):
    """
    Calls the existing ML model (score_single_lead).
    Returns real score, probability, priority, and explanation.
    """
    try:
        lead_dict = lead_input.model_dump()
        result = score_single_lead(lead_dict)

        score = int(result["lead_score"])
        prob = float(result["conversion_probability"])
        priority = str(result["priority"])     # High / Medium / Low (from model)
        category = str(result["lead_priority"]).upper()  # HOT / WARM / COLD
        if category not in ("HOT", "WARM", "COLD"):
            category = "HOT" if prob >= 0.75 else ("WARM" if prob >= 0.40 else "COLD")

        explanation = generate_score_explanation(lead_dict, score, prob)

        return {
            "success": True,
            "score": score,
            "conversion_probability": round(prob, 4),
            "conversion_probability_pct": round(prob * 100, 1),
            "category": category,
            "priority": priority,
            "explanation": explanation,
            "input_data": lead_dict,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ML prediction error: {str(e)}")


@app.get("/api/leads")
def get_leads(
    search: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    pipeline_stage: Optional[str] = Query(None),
    industry: Optional[str] = Query(None),
    lead_source: Optional[str] = Query(None),
    assigned_to: Optional[str] = Query(None),
    sort_by: str = Query("score"),
    sort_dir: str = Query("desc"),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
):
    leads = supabase_db.get_all_leads(
        search=search, status=status, category=category,
        pipeline_stage=pipeline_stage, assigned_to=assigned_to,
        industry=industry, lead_source=lead_source,
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
def create_lead(lead: CreateLeadSchema):
    try:
        created = supabase_db.create_lead(lead.model_dump())
        return {"success": True, "lead": created}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/leads/{lead_id}")
def update_lead(lead_id: str, updates: UpdateLeadSchema):
    data = {k: v for k, v in updates.model_dump().items() if v is not None}
    if not data:
        raise HTTPException(status_code=400, detail="No update fields provided")
    updated = supabase_db.update_lead(lead_id, data)
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
def add_note(lead_id: str, note_in: NoteInput):
    if not note_in.note.strip():
        raise HTTPException(status_code=400, detail="Note cannot be empty")
    note = supabase_db.add_lead_note(lead_id, note_in.note.strip(), note_in.author)
    return {"success": True, "note": note}


@app.get("/api/leads/{lead_id}/activities")
def get_activities(lead_id: str):
    return {"success": True, "activities": supabase_db.get_lead_activities(lead_id)}


@app.get("/api/dashboard")
def get_dashboard():
    summary = supabase_db.get_dashboard_summary()
    return {"success": True, "dashboard": summary}


@app.get("/api/analytics")
def get_analytics():
    data = supabase_db.get_analytics()
    return {"success": True, "analytics": data}


@app.post("/api/import")
async def import_csv(file: UploadFile = File(...)):
    """
    CSV Import endpoint.
    Validates columns, scores new leads via ML model, detects duplicates,
    and imports into Supabase.
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

    if not rows:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    headers_lower = [h.lower().strip() for h in reader.fieldnames or []]

    # Validate minimum required columns
    required = {"name", "email"}
    missing_req = required - set(headers_lower)
    if missing_req:
        raise HTTPException(
            status_code=422,
            detail=f"CSV is missing required columns: {', '.join(missing_req)}"
        )

    # Column name mapping (case-insensitive aliases)
    col_map = {h: h for h in headers_lower}

    def get_col(row, *aliases):
        for alias in aliases:
            if alias in row:
                return row.get(alias, "")
            # try from col_map
            for k in row:
                if k.lower().strip() == alias:
                    return row[k]
        return ""

    records_to_import = []
    invalid_rows = 0

    for i, row in enumerate(rows):
        row_lower = {k.lower().strip(): v.strip() for k, v in row.items() if k}

        name = get_col(row_lower, "name") or f"CSV Lead {i+1}"
        email = get_col(row_lower, "email")
        if not email:
            invalid_rows += 1
            continue

        company_size_raw = get_col(row_lower, "company_size", "companysize")
        try:
            company_size = float(company_size_raw) if company_size_raw else 100.0
        except ValueError:
            company_size = 100.0

        target_raw = get_col(row_lower, "target", "converted", "is_converted")
        converted = None
        if target_raw.lower() in ("1", "true", "yes", "won"):
            converted = True
        elif target_raw.lower() in ("0", "false", "no", "lost"):
            converted = False

        rec = {
            "name": name,
            "email": email,
            "phone": get_col(row_lower, "phone"),
            "company": get_col(row_lower, "company"),
            "job_title": get_col(row_lower, "job_title", "jobtitle"),
            "industry": get_col(row_lower, "industry") or "Unknown",
            "company_size": company_size,
            "location": get_col(row_lower, "location"),
            "lead_source": get_col(row_lower, "lead_source", "leadsource", "source") or "Import",
            "product_interest": get_col(row_lower, "product_interest") or "",
            "budget_range": get_col(row_lower, "budget_range") or "Unknown",
            "website_visits": float(get_col(row_lower, "website_visits") or 0),
            "page_views": float(get_col(row_lower, "page_views") or 0),
            "pricing_page_visits": float(get_col(row_lower, "pricing_page_visits") or 0),
            "demo_requested": get_col(row_lower, "demo_requested") or "No",
            "email_opens": float(get_col(row_lower, "email_opens") or 0),
            "form_completions": float(get_col(row_lower, "form_completions") or 0),
            "content_downloads": float(get_col(row_lower, "content_downloads") or 0),
            "previous_interactions": float(get_col(row_lower, "previous_interactions") or 0),
            "response_time_hours": float(get_col(row_lower, "response_time_hours") or 0),
            "num_calls": float(get_col(row_lower, "num_calls") or 0),
            "num_meetings": float(get_col(row_lower, "num_meetings") or 0),
            "status": get_col(row_lower, "status") or "New",
            "pipeline_stage": get_col(row_lower, "pipeline_stage") or "New",
            "converted": converted,
            "outcome": "Won" if converted is True else ("Lost" if converted is False else None),
        }

        # Score via existing ML model
        try:
            ml_result = score_single_lead({
                "industry": rec["industry"],
                "company_size": rec["company_size"],
                "lead_source": rec["lead_source"],
                "product_interest": rec["product_interest"],
                "budget_range": rec["budget_range"],
                "website_visits": rec["website_visits"],
                "page_views": rec["page_views"],
                "pricing_page_visits": rec["pricing_page_visits"],
                "demo_requested": rec["demo_requested"],
                "email_opens": rec["email_opens"],
                "form_completions": rec["form_completions"],
                "content_downloads": rec["content_downloads"],
                "previous_interactions": rec["previous_interactions"],
                "response_time_hours": rec["response_time_hours"],
                "num_calls": rec["num_calls"],
                "num_meetings": rec["num_meetings"],
            })
            rec["score"] = int(ml_result["lead_score"])
            rec["conversion_probability"] = round(float(ml_result["conversion_probability"]), 4)
            rec["priority"] = str(ml_result["priority"])
            prob = float(ml_result["conversion_probability"])
            rec["category"] = "HOT" if prob >= 0.75 else ("WARM" if prob >= 0.40 else "COLD")
        except Exception:
            rec["score"] = 50
            rec["conversion_probability"] = 0.50
            rec["category"] = "WARM"
            rec["priority"] = "Medium"

        records_to_import.append(rec)

    result = supabase_db.import_csv_records(records_to_import)
    total_rows = len(rows)

    return {
        "success": True,
        "filename": file.filename,
        "total_rows": total_rows,
        "valid_rows": len(records_to_import),
        "invalid_rows": invalid_rows,
        "imported": result["imported"],
        "duplicates": result["duplicates"],
        "failed": result["failed"],
    }


@app.get("/api/pipeline-stages")
def get_pipeline_stages():
    return {
        "success": True,
        "stages": supabase_db.PIPELINE_STAGES,
    }


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
