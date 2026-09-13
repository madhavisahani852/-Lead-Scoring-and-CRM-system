import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Lead Scoring" in data["service"]

def test_score_lead_endpoint():
    payload = {
        "name": "Test Lead",
        "company": "Test Enterprise",
        "industry": "SaaS",
        "company_size": 250,
        "lead_source": "Website",
        "product_interest": "Enterprise Plan",
        "budget_range": "50k-1L",
        "website_visits": 15,
        "page_views": 45,
        "pricing_page_visits": 6,
        "demo_requested": "Yes",
        "email_opens": 10,
        "form_completions": 2,
        "content_downloads": 3,
        "previous_interactions": 4,
        "response_time_hours": 2.5,
        "num_calls": 3,
        "num_meetings": 1
    }
    response = client.post("/api/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert 0 <= data["lead_score"] <= 100
    assert data["priority"] in ["High", "Medium", "Low"]
    assert "recommended_action" in data
    assert len(data["key_drivers"]) > 0

def test_get_leads_endpoint():
    response = client.get("/api/leads?page=1&limit=10")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["total"] > 0
    assert len(data["leads"]) == 10
    first_lead = data["leads"][0]
    assert "lead_id" in first_lead
    assert "lead_score" in first_lead
    assert "priority" in first_lead

def test_get_analytics_endpoint():
    response = client.get("/api/analytics")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "kpis" in data
    assert data["kpis"]["total_leads"] > 0
    assert "score_distribution" in data
    assert len(data["score_distribution"]) == 5

def test_get_model_info_endpoint():
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "model_name" in data
    assert "test_metrics" in data
    assert "top_features" in data

def test_get_sample_leads_endpoint():
    response = client.get("/api/sample-leads")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["samples"]) >= 3
