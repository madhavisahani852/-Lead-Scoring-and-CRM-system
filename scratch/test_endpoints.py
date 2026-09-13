import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def run_tests():
    print("==================================================")
    print("RUNNING API ENDPOINT VERIFICATION TESTS")
    print("==================================================")

    # 1. Health Check
    r = client.get("/api/health")
    print("1. GET /api/health ->", r.status_code, r.json())
    assert r.status_code == 200, "Health check failed"

    # 2. Dashboard
    r = client.get("/api/dashboard")
    print("2. GET /api/dashboard ->", r.status_code, "Total leads:", r.json()["dashboard"]["total_leads"])
    assert r.status_code == 200, "Dashboard failed"

    # 3. Score Lead (ML Bridge)
    score_payload = {
        "name": "Test Prospect",
        "email": "test@prospect.com",
        "company": "Test Co",
        "industry": "SaaS",
        "company_size": 250,
        "lead_source": "Website",
        "product_interest": "Enterprise Plan",
        "budget_range": "50k-1L",
        "website_visits": 12,
        "page_views": 35,
        "pricing_page_visits": 5,
        "demo_requested": "Yes",
        "email_opens": 8,
        "form_completions": 2,
        "content_downloads": 3,
        "previous_interactions": 4,
        "response_time_hours": 2.5,
        "num_calls": 3,
        "num_meetings": 1
    }
    r = client.post("/api/score", json=score_payload)
    print("3. POST /api/score ->", r.status_code, r.json())
    assert r.status_code == 200 and "score" in r.json(), "Score endpoint failed"
    score_val = r.json()["score"]
    category_val = r.json()["category"]
    print(f"   ML Model Output: Score={score_val}, Category={category_val}")

    # 4. Create Lead
    new_lead_payload = {
        **score_payload,
        "score": score_val,
        "category": category_val,
        "status": "New"
    }
    r = client.post("/api/leads", json=new_lead_payload)
    print("4. POST /api/leads ->", r.status_code, "Created Lead ID:", r.json()["lead"]["id"])
    assert r.status_code == 200, "Create lead failed"
    created_id = r.json()["lead"]["id"]

    # 5. Get Leads (Filtered)
    r = client.get(f"/api/leads?search={created_id}")
    print("5. GET /api/leads -> Found leads count:", len(r.json()["leads"]))
    assert r.status_code == 200 and len(r.json()["leads"]) >= 1, "Get leads failed"

    # 6. Get Single Lead
    r = client.get(f"/api/leads/{created_id}")
    print("6. GET /api/leads/{id} ->", r.status_code, r.json()["lead"]["name"])
    assert r.status_code == 200, "Get single lead failed"

    # 7. Update Lead Status
    r = client.put(f"/api/leads/{created_id}", json={"status": "Contacted"})
    print("7. PUT /api/leads/{id} -> Updated Status:", r.json()["lead"]["status"])
    assert r.status_code == 200 and r.json()["lead"]["status"] == "Contacted", "Update lead failed"

    # 8. Add Note
    r = client.post(f"/api/leads/{created_id}/notes", json={"note": "Test note entry"})
    print("8. POST /api/leads/{id}/notes -> Added Note:", r.json()["note"]["note"])
    assert r.status_code == 200, "Add note failed"

    # 9. Get Notes
    r = client.get(f"/api/leads/{created_id}/notes")
    print("9. GET /api/leads/{id}/notes -> Notes count:", len(r.json()["notes"]))
    assert r.status_code == 200 and len(r.json()["notes"]) >= 1, "Get notes failed"

    # 10. Delete Lead
    r = client.delete(f"/api/leads/{created_id}")
    print("10. DELETE /api/leads/{id} ->", r.status_code, r.json())
    assert r.status_code == 200, "Delete lead failed"

    print("==================================================")
    print("ALL API ENDPOINTS PASSED VERIFICATION TEST SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
