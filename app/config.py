import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"
MODEL_DIR = APP_DIR / "ml_models"
DATABASE_PATH = BASE_DIR / "capitalblue_fastapi.db"

load_dotenv(BASE_DIR / ".env")

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
SESSION_SECRET = os.getenv("SESSION_SECRET", "capitalblue-fastapi-session-secret")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
