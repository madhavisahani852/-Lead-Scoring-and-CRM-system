import sys
import webbrowser
import time
import uvicorn
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("=" * 60)
    print("STARTING LEAD SCORING & CRM WEB APPLICATION")
    print("=" * 60)
    print("Local Server Address: http://127.0.0.1:8000")
    print("API Documentation:    http://127.0.0.1:8000/docs")
    print("Model Artifact:       Best Tuned XGBoost Pipeline")
    print("Cleaned Dataset:      Loaded")
    print("=" * 60)

    # Launch FastAPI application via uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()
