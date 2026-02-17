from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from config import DB_PATH, DEFAULT_TREND_TERMS, DEFAULT_FUZZY_THRESHOLD, ensure_data_dir


CREATE_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS papers (
        arxiv_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        abstract TEXT NOT NULL,
        authors_json TEXT NOT NULL,
        published_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        primary_category TEXT NOT NULL,
        all_categories_json TEXT NOT NULL,
        doi TEXT,
        entry_url TEXT NOT NULL,
        pdf_url TEXT,
        has_code INTEGER NOT NULL DEFAULT 0,
        github_links_json TEXT NOT NULL DEFAULT '[]',
        ingested_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS paper_metrics (
        arxiv_id TEXT PRIMARY KEY,
        citation_count INTEGER DEFAULT 0,
        citations_12m INTEGER DEFAULT 0,
        citation_velocity REAL DEFAULT 0,
        impact_score REAL DEFAULT 0,
        frontier_score REAL DEFAULT 0,
        github_stars_max INTEGER DEFAULT 0,
        open_source_score REAL DEFAULT 0,
        keyword_burst REAL DEFAULT 0,
        freshness REAL DEFAULT 0,
        openalex_id TEXT,
        metrics_updated_at TEXT,
        FOREIGN KEY(arxiv_id) REFERENCES papers(arxiv_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS paper_keywords (
        arxiv_id TEXT NOT NULL,
        keyword TEXT NOT NULL,
        score REAL NOT NULL,
        source TEXT NOT NULL CHECK(source IN ('tfidf', 'manual', 'alias')),
        PRIMARY KEY (arxiv_id, keyword, source),
        FOREIGN KEY(arxiv_id) REFERENCES papers(arxiv_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pipeline_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tracked_terms (
        term TEXT PRIMARY KEY,
        aliases_json TEXT NOT NULL DEFAULT '[]',
        fuzzy_threshold INTEGER NOT NULL DEFAULT 85,
        is_active INTEGER NOT NULL DEFAULT 1
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
        arxiv_id UNINDEXED,
        title,
        abstract
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_papers_published_at ON papers(published_at)",
    "CREATE INDEX IF NOT EXISTS idx_papers_primary_category ON papers(primary_category)",
    "CREATE INDEX IF NOT EXISTS idx_papers_has_code ON papers(has_code)",
    "CREATE INDEX IF NOT EXISTS idx_keywords_keyword ON paper_keywords(keyword)",
    "CREATE INDEX IF NOT EXISTS idx_metrics_frontier ON paper_metrics(frontier_score)",
    "CREATE INDEX IF NOT EXISTS idx_metrics_updated_at ON paper_metrics(metrics_updated_at)",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    ensure_data_dir()
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -200000")
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    with conn:
        for stmt in CREATE_STATEMENTS:
            conn.execute(stmt)
    seed_default_terms(conn)


def seed_default_terms(conn: sqlite3.Connection) -> None:
    rows = [
        ("Diffusion", json.dumps(["diffusion model", "ddpm", "score-based"])),
        ("LLM", json.dumps(["large language model", "gpt", "foundation model"])),
        ("Agent", json.dumps(["autonomous agent", "multi-agent", "tool use"])),
    ]
    with conn:
        for term, aliases in rows:
            conn.execute(
                """
                INSERT OR IGNORE INTO tracked_terms(term, aliases_json, fuzzy_threshold, is_active)
                VALUES (?, ?, ?, 1)
                """,
                (term, aliases, DEFAULT_FUZZY_THRESHOLD),
            )
        for term in DEFAULT_TREND_TERMS:
            conn.execute(
                """
                INSERT OR IGNORE INTO tracked_terms(term, aliases_json, fuzzy_threshold, is_active)
                VALUES (?, '[]', ?, 1)
                """,
                (term, DEFAULT_FUZZY_THRESHOLD),
            )


def get_state(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM pipeline_state WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    return row["value"]


def set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    with conn:
        conn.execute(
            """
            INSERT INTO pipeline_state(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )


def _to_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True)


def upsert_papers(conn: sqlite3.Connection, papers: Iterable[dict[str, Any]]) -> int:
    records = list(papers)
    if not records:
        return 0

    sql = """
    INSERT INTO papers (
        arxiv_id, title, abstract, authors_json, published_at, updated_at,
        primary_category, all_categories_json, doi, entry_url, pdf_url,
        has_code, github_links_json, ingested_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(arxiv_id) DO UPDATE SET
        title = excluded.title,
        abstract = excluded.abstract,
        authors_json = excluded.authors_json,
        published_at = excluded.published_at,
        updated_at = excluded.updated_at,
        primary_category = excluded.primary_category,
        all_categories_json = excluded.all_categories_json,
        doi = excluded.doi,
        entry_url = excluded.entry_url,
        pdf_url = excluded.pdf_url,
        has_code = excluded.has_code,
        github_links_json = excluded.github_links_json,
        ingested_at = excluded.ingested_at
    """

    with conn:
        for rec in records:
            conn.execute(
                sql,
                (
                    rec["arxiv_id"],
                    rec["title"],
                    rec["abstract"],
                    _to_json(rec["authors_json"]),
                    rec["published_at"],
                    rec["updated_at"],
                    rec["primary_category"],
                    _to_json(rec["all_categories_json"]),
                    rec.get("doi"),
                    rec["entry_url"],
                    rec.get("pdf_url"),
                    int(bool(rec.get("has_code", False))),
                    _to_json(rec.get("github_links_json", [])),
                    rec.get("ingested_at", utc_now_iso()),
                ),
            )
            conn.execute("DELETE FROM papers_fts WHERE arxiv_id = ?", (rec["arxiv_id"],))
            conn.execute(
                "INSERT INTO papers_fts(arxiv_id, title, abstract) VALUES (?, ?, ?)",
                (rec["arxiv_id"], rec["title"], rec["abstract"]),
            )
    return len(records)


def upsert_metrics(conn: sqlite3.Connection, metrics: Iterable[dict[str, Any]]) -> int:
    rows = list(metrics)
    if not rows:
        return 0

    sql = """
    INSERT INTO paper_metrics (
        arxiv_id, citation_count, citations_12m, citation_velocity, impact_score,
        frontier_score, github_stars_max, open_source_score, keyword_burst, freshness,
        openalex_id, metrics_updated_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(arxiv_id) DO UPDATE SET
        citation_count = excluded.citation_count,
        citations_12m = excluded.citations_12m,
        citation_velocity = excluded.citation_velocity,
        impact_score = excluded.impact_score,
        frontier_score = excluded.frontier_score,
        github_stars_max = excluded.github_stars_max,
        open_source_score = excluded.open_source_score,
        keyword_burst = excluded.keyword_burst,
        freshness = excluded.freshness,
        openalex_id = excluded.openalex_id,
        metrics_updated_at = excluded.metrics_updated_at
    """

    with conn:
        for row in rows:
            conn.execute(
                sql,
                (
                    row["arxiv_id"],
                    int(row.get("citation_count", 0) or 0),
                    int(row.get("citations_12m", 0) or 0),
                    float(row.get("citation_velocity", 0.0) or 0.0),
                    float(row.get("impact_score", 0.0) or 0.0),
                    float(row.get("frontier_score", 0.0) or 0.0),
                    int(row.get("github_stars_max", 0) or 0),
                    float(row.get("open_source_score", 0.0) or 0.0),
                    float(row.get("keyword_burst", 0.0) or 0.0),
                    float(row.get("freshness", 0.0) or 0.0),
                    row.get("openalex_id"),
                    row.get("metrics_updated_at", utc_now_iso()),
                ),
            )
    return len(rows)


def replace_tfidf_keywords(conn: sqlite3.Connection, rows: Iterable[tuple[str, str, float, str]]) -> None:
    with conn:
        conn.execute("DELETE FROM paper_keywords WHERE source = 'tfidf'")
        conn.executemany(
            """
            INSERT INTO paper_keywords(arxiv_id, keyword, score, source)
            VALUES (?, ?, ?, ?)
            """,
            list(rows),
        )


def replace_alias_keywords(conn: sqlite3.Connection, rows: Iterable[tuple[str, str, float, str]]) -> None:
    with conn:
        conn.execute("DELETE FROM paper_keywords WHERE source = 'alias'")
        conn.executemany(
            """
            INSERT INTO paper_keywords(arxiv_id, keyword, score, source)
            VALUES (?, ?, ?, ?)
            """,
            list(rows),
        )


def get_papers_for_enrichment(conn: sqlite3.Connection, limit: int = 300) -> list[sqlite3.Row]:
    sql = """
    SELECT
        p.arxiv_id,
        p.github_links_json,
        p.published_at,
        m.metrics_updated_at
    FROM papers p
    LEFT JOIN paper_metrics m ON m.arxiv_id = p.arxiv_id
    WHERE p.has_code = 1
      AND (
           m.arxiv_id IS NULL
        OR m.metrics_updated_at IS NULL
       OR datetime(m.metrics_updated_at) < datetime('now', '-14 day')
      )
    ORDER BY p.published_at DESC
    LIMIT ?
    """
    return list(conn.execute(sql, (limit,)).fetchall())


def get_papers_for_enrichment_by_ids(conn: sqlite3.Connection, arxiv_ids: Iterable[str]) -> list[sqlite3.Row]:
    ids = [str(v) for v in arxiv_ids if str(v).strip()]
    if not ids:
        return []

    rows: list[sqlite3.Row] = []
    chunk_size = 900
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start : start + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        sql = f"""
        SELECT
            p.arxiv_id,
            p.github_links_json,
            p.published_at,
            m.metrics_updated_at
        FROM papers p
        LEFT JOIN paper_metrics m ON m.arxiv_id = p.arxiv_id
        WHERE p.arxiv_id IN ({placeholders})
          AND p.has_code = 1
          AND (
               m.arxiv_id IS NULL
            OR m.metrics_updated_at IS NULL
           OR datetime(m.metrics_updated_at) < datetime('now', '-14 day')
          )
        ORDER BY p.published_at DESC
        """
        rows.extend(conn.execute(sql, chunk).fetchall())
    return rows


def get_active_tracked_terms(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT term, aliases_json, fuzzy_threshold
            FROM tracked_terms
            WHERE is_active = 1
            ORDER BY term
            """
        ).fetchall()
    )


def list_categories(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT primary_category
        FROM papers
        WHERE primary_category IS NOT NULL
        ORDER BY primary_category
        """
    ).fetchall()
    return [row["primary_category"] for row in rows]


def get_latest_published_at(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT MAX(published_at) AS latest_published_at FROM papers").fetchone()
    if row is None:
        return None
    return row["latest_published_at"]


def fetch_existing_arxiv_ids(conn: sqlite3.Connection, arxiv_ids: Iterable[str]) -> set[str]:
    ids = [str(v) for v in arxiv_ids if str(v).strip()]
    if not ids:
        return set()

    existing: set[str] = set()
    chunk_size = 900
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start : start + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        sql = f"SELECT arxiv_id FROM papers WHERE arxiv_id IN ({placeholders})"
        rows = conn.execute(sql, chunk).fetchall()
        existing.update(str(row["arxiv_id"]) for row in rows)
    return existing


def prune_oldest_papers(conn: sqlite3.Connection, limit: int) -> int:
    n = int(limit or 0)
    if n <= 0:
        return 0
    rows = conn.execute(
        """
        SELECT arxiv_id
        FROM papers
        ORDER BY published_at ASC, arxiv_id ASC
        LIMIT ?
        """,
        (n,),
    ).fetchall()
    arxiv_ids = [str(row["arxiv_id"]) for row in rows]
    if not arxiv_ids:
        return 0

    chunk_size = 900
    with conn:
        for start in range(0, len(arxiv_ids), chunk_size):
            chunk = arxiv_ids[start : start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            conn.execute(f"DELETE FROM papers_fts WHERE arxiv_id IN ({placeholders})", chunk)
            conn.execute(f"DELETE FROM papers WHERE arxiv_id IN ({placeholders})", chunk)
    return len(arxiv_ids)
