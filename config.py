from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("ARXIV_TRACKER_DATA_DIR", ROOT_DIR / "data"))
DB_PATH = Path(os.getenv("ARXIV_TRACKER_DB_PATH", DATA_DIR / "arxiv_insight.db"))

DEFAULT_CATEGORIES = ["cs.AI", "cs.LG", "cs.CV", "cs.CL"]
DEFAULT_CROSS_DOMAIN_PREFIXES = ["physics", "q-bio", "math"]
DEFAULT_TREND_TERMS = ["Diffusion", "LLM", "Agent"]

ARXIV_DELAY_SECONDS = float(os.getenv("ARXIV_DELAY_SECONDS", "3.0"))
ARXIV_NUM_RETRIES = int(os.getenv("ARXIV_NUM_RETRIES", "5"))
ARXIV_PAGE_SIZE = int(os.getenv("ARXIV_PAGE_SIZE", "100"))
ARXIV_INTER_CATEGORY_SLEEP = float(os.getenv("ARXIV_INTER_CATEGORY_SLEEP", "1.0"))

GITHUB_API_BASE_URL = os.getenv("GITHUB_API_BASE_URL", "https://api.github.com")
GITHUB_TOKEN = os.getenv(
    "GITHUB_TOKEN",
    "",  # use your own token
)
GITHUB_DELAY_SECONDS = float(os.getenv("GITHUB_DELAY_SECONDS", "0.5"))

REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))
ENRICH_HTTP_MAX_RETRIES = int(os.getenv("ENRICH_HTTP_MAX_RETRIES", "4"))
DEFAULT_FUZZY_THRESHOLD = int(os.getenv("DEFAULT_FUZZY_THRESHOLD", "85"))

FRONTIER_KEYWORD_WEIGHT = float(os.getenv("FRONTIER_KEYWORD_WEIGHT", "0.55"))
FRONTIER_OPEN_SOURCE_WEIGHT = float(os.getenv("FRONTIER_OPEN_SOURCE_WEIGHT", "0.30"))
FRONTIER_FRESHNESS_WEIGHT = float(os.getenv("FRONTIER_FRESHNESS_WEIGHT", "0.15"))

OPEN_SOURCE_HAS_CODE_WEIGHT = float(os.getenv("OPEN_SOURCE_HAS_CODE_WEIGHT", "0.7"))
OPEN_SOURCE_STAR_WEIGHT = float(os.getenv("OPEN_SOURCE_STAR_WEIGHT", "0.3"))

KEYWORD_BURST_RECENT_MONTHS = int(os.getenv("KEYWORD_BURST_RECENT_MONTHS", "6"))
KEYWORD_BURST_BASELINE_MONTHS = int(os.getenv("KEYWORD_BURST_BASELINE_MONTHS", "18"))

TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", "2000"))
TFIDF_TOP_K = int(os.getenv("TFIDF_TOP_K", "8"))


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
