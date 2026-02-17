from __future__ import annotations

from datetime import datetime, timezone

from db import get_connection, init_db, prune_oldest_papers, upsert_papers
from ingest import build_backfill_windows
from processor import query_papers


def test_upsert_papers_is_idempotent(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    record = {
        "arxiv_id": "1234.5678",
        "title": "Paper A",
        "abstract": "A short abstract",
        "authors_json": ["Alice", "Bob"],
        "published_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
        "primary_category": "cs.AI",
        "all_categories_json": ["cs.AI", "cs.LG"],
        "doi": None,
        "entry_url": "http://arxiv.org/abs/1234.5678",
        "pdf_url": "http://arxiv.org/pdf/1234.5678",
        "has_code": True,
        "github_links_json": ["https://github.com/org/repo"],
        "ingested_at": "2025-01-01T01:00:00+00:00",
    }
    upsert_papers(conn, [record])
    record["title"] = "Paper A Updated"
    upsert_papers(conn, [record])

    row = conn.execute("SELECT COUNT(*) AS c FROM papers").fetchone()
    assert row["c"] == 1
    updated = conn.execute("SELECT title FROM papers WHERE arxiv_id = ?", ("1234.5678",)).fetchone()
    assert updated["title"] == "Paper A Updated"


def test_build_backfill_windows_monthly_chunks():
    now = datetime(2026, 2, 14, tzinfo=timezone.utc)
    windows = build_backfill_windows(["cs.AI"], years=1, now=now)
    # About 12 monthly chunks for one year.
    assert 11 <= len(windows) <= 13
    assert all(item[0] == "cs.AI" for item in windows)


def test_prune_oldest_papers_removes_oldest_and_fts(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    base = {
        "title": "Paper",
        "abstract": "A short abstract",
        "authors_json": ["Alice"],
        "updated_at": "2025-01-01T00:00:00+00:00",
        "primary_category": "cs.AI",
        "all_categories_json": ["cs.AI"],
        "doi": None,
        "entry_url": "http://arxiv.org/abs/0",
        "pdf_url": "http://arxiv.org/pdf/0",
        "has_code": False,
        "github_links_json": [],
        "ingested_at": "2025-01-01T01:00:00+00:00",
    }
    rows = []
    for idx, published_at in enumerate(
        [
            "2024-01-01T00:00:00+00:00",
            "2024-02-01T00:00:00+00:00",
            "2024-03-01T00:00:00+00:00",
        ]
    ):
        row = dict(base)
        row["arxiv_id"] = f"1234.56{idx}"
        row["published_at"] = published_at
        row["entry_url"] = f"http://arxiv.org/abs/1234.56{idx}"
        row["pdf_url"] = f"http://arxiv.org/pdf/1234.56{idx}"
        rows.append(row)
    upsert_papers(conn, rows)

    deleted = prune_oldest_papers(conn, 2)
    assert deleted == 2
    count_row = conn.execute("SELECT COUNT(*) AS c FROM papers").fetchone()
    assert count_row["c"] == 1
    remain = conn.execute("SELECT arxiv_id FROM papers").fetchone()
    assert remain["arxiv_id"] == "1234.562"
    fts_row = conn.execute("SELECT COUNT(*) AS c FROM papers_fts").fetchone()
    assert fts_row["c"] == 1


def test_query_papers_respects_datetime_window_and_has_code_only(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    base = {
        "title": "Paper",
        "abstract": "A short abstract",
        "authors_json": ["Alice"],
        "updated_at": "2025-01-01T00:00:00+00:00",
        "primary_category": "cs.AI",
        "all_categories_json": ["cs.AI"],
        "doi": None,
        "has_code": False,
        "github_links_json": [],
        "ingested_at": "2025-01-01T01:00:00+00:00",
    }
    rows = []
    for idx, ts in enumerate(
        [
            "2025-01-01T10:00:00+00:00",
            "2025-01-01T15:00:00+00:00",
            "2025-01-01T20:00:00+00:00",
        ]
    ):
        row = dict(base)
        row["arxiv_id"] = f"9999.00{idx}"
        row["title"] = f"Paper {idx}"
        row["published_at"] = ts
        row["entry_url"] = f"http://arxiv.org/abs/9999.00{idx}"
        row["pdf_url"] = f"http://arxiv.org/pdf/9999.00{idx}"
        row["has_code"] = idx == 1
        if idx == 1:
            row["github_links_json"] = ["https://github.com/org/repo"]
        rows.append(row)
    upsert_papers(conn, rows)

    start = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 1, 1, 18, 0, 0, tzinfo=timezone.utc)
    out = query_papers(
        conn,
        start_date=start,
        end_date=end,
        categories=["cs.AI"],
        sort_key="newest",
    )
    assert out["arxiv_id"].tolist() == ["9999.001"]

    out_code = query_papers(
        conn,
        start_date=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        end_date=datetime(2025, 1, 1, 23, 0, 0, tzinfo=timezone.utc),
        categories=["cs.AI"],
        has_code_only=True,
        sort_key="newest",
    )
    assert out_code["arxiv_id"].tolist() == ["9999.001"]
