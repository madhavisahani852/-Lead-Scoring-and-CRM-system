import sys
from pathlib import Path

# Ensure project root is in sys.path for standalone script execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, Union
import pandas as pd
import json
from ml.inference.predict import predict_batch


def score_single_lead(lead_dict: Dict[str, Any], model_path: str = None) -> Dict[str, Any]:
    """
    Scores a single lead input dictionary and returns conversion probability,
    lead score (0-100), binary prediction, and priority tier.
    """
    lead_df = pd.DataFrame([lead_dict])
    scored_df = predict_batch(lead_df, model_path=model_path)
    result_row = scored_df.iloc[0]

    return {
        "lead_id": lead_dict.get("lead_id", "N/A"),
        "conversion_probability": float(result_row["conversion_probability"]),
        "lead_score": int(result_row["lead_score"]),
        "predicted_conversion": int(result_row["predicted_conversion"]),
        "lead_priority": str(result_row["lead_priority"]),
    }


def main():
    """
    Demonstration CLI runner for single lead scoring.
    """
    sample_lead = {
        "lead_id": "L9999",
        "company_size": 250,
        "industry": "SaaS",
        "lead_source": "Website",
        "product_interest": "Enterprise",
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
        "num_meetings": 1,
    }

    print("=" * 60)
    print("SINGLE LEAD SCORING INFERENCE DEMO")
    print("=" * 60)
    print("Input Lead Data:")
    print(json.dumps(sample_lead, indent=2))

    try:
        result = score_single_lead(sample_lead)
        print("\nScoring Output:")
        print(json.dumps(result, indent=2))
        print("=" * 60)
    except Exception as e:
        print(f"\nInference Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
