import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_auth_workflow():
    # 1. Signup
    signup_payload = {
        "email": "sarah.connor@test.com",
        "password": "Password123!",
        "name": "Sarah Connor"
    }
    res = client.post("/api/auth/signup", json=signup_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "token" in data
    assert data["user"]["email"] == "sarah.connor@test.com"
    token = data["token"]

    # 2. Check me with token
    res_me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res_me.status_code == 200
    me_data = res_me.json()
    assert me_data["authenticated"] is True
    assert me_data["user"]["email"] == "sarah.connor@test.com"

    # 3. Login
    login_payload = {
        "email": "sarah.connor@test.com",
        "password": "Password123!"
    }
    res_login = client.post("/api/auth/login", json=login_payload)
    assert res_login.status_code == 200
    login_data = res_login.json()
    assert login_data["success"] is True
    assert "token" in login_data

    # 4. Logout
    res_logout = client.post("/api/auth/logout")
    assert res_logout.status_code == 200
    assert res_logout.json()["success"] is True


def test_workspace_and_members():
    # Get workspace
    res = client.get("/api/workspace")
    assert res.status_code == 200
    assert res.json()["success"] is True
    assert "workspace" in res.json()

    # Get members
    res_mem = client.get("/api/workspace/members")
    assert res_mem.status_code == 200
    members = res_mem.json()["members"]
    assert len(members) >= 1
    assert any("name" in m for m in members)

    # Add member
    new_mem = {
        "email": "john.reese@test.com",
        "name": "John Reese",
        "role": "Sales Rep"
    }
    res_add = client.post("/api/workspace/members", json=new_mem)
    assert res_add.status_code == 200
    assert res_add.json()["member"]["email"] == "john.reese@test.com"


def test_lead_assignment_and_activities():
    # 1. Create lead
    lead_payload = {
        "name": "Assignment Test Corp",
        "email": "assignee@testcorp.com",
        "industry": "SaaS",
        "company_size": 350,
        "lead_source": "Website",
        "product_interest": "Enterprise Plan",
        "budget_range": "50k-1L",
        "score": 85,
        "conversion_probability": 0.85,
        "pipeline_stage": "New"
    }
    create_res = client.post("/api/leads", json=lead_payload)
    assert create_res.status_code == 200
    lead_id = create_res.json()["lead"]["id"]

    # 2. Assign lead to Sarah Jenkins
    update_res = client.put(f"/api/leads/{lead_id}", json={"assigned_to": "Sarah Jenkins"})
    assert update_res.status_code == 200
    assert update_res.json()["lead"]["assigned_to"] == "Sarah Jenkins"

    # 3. Verify LEAD_ASSIGNED activity
    act_res = client.get(f"/api/leads/{lead_id}/activities")
    assert act_res.status_code == 200
    activities = act_res.json()["activities"]
    types = [a["activity_type"] for a in activities]
    assert "LEAD_CREATED" in types
    assert "LEAD_ASSIGNED" in types


def test_historical_won_lost_tracking():
    lead_payload = {
        "name": "Deal Outcome Test",
        "email": "deal@outcometest.com",
        "score": 90,
        "pipeline_stage": "Proposal"
    }
    lead_id = client.post("/api/leads", json=lead_payload).json()["lead"]["id"]

    # Move to Won
    won_res = client.put(f"/api/leads/{lead_id}", json={"pipeline_stage": "Won"})
    assert won_res.status_code == 200
    won_lead = won_res.json()["lead"]
    assert won_lead["converted"] is True
    assert won_lead["outcome"] == "Won"
    assert won_lead["converted_at"] is not None

    # Verify WON activity
    act_res = client.get(f"/api/leads/{lead_id}/activities")
    activities = act_res.json()["activities"]
    assert any(a["activity_type"] == "WON" for a in activities)


def test_smart_alerts():
    res = client.get("/api/alerts")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "alerts" in data
    assert isinstance(data["alerts"], list)
    if data["alerts"]:
        alert = data["alerts"][0]
        assert "type" in alert
        assert "title" in alert
        assert "severity" in alert


def test_csv_export():
    res = client.get("/api/export")
    assert res.status_code == 200
    assert "text/csv" in res.headers["content-type"]
    assert "name,email" in res.text or "id,name" in res.text
    lines = res.text.strip().split("\n")
    assert len(lines) > 1


def test_model_versioning_and_retraining():
    # 1. Model versions
    res_versions = client.get("/api/model/versions")
    assert res_versions.status_code == 200
    versions = res_versions.json()["versions"]
    assert len(versions) >= 1
    assert any(v["status"] == "Production" for v in versions)

    # 2. Retrain endpoint
    res_retrain = client.post("/api/model/retrain")
    assert res_retrain.status_code == 200
    retrain_data = res_retrain.json()
    assert retrain_data["success"] is True
    assert "promoted" in retrain_data
    assert "candidate_metrics" in retrain_data
    assert "roc_auc" in retrain_data["candidate_metrics"]
