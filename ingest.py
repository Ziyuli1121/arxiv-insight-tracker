from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Iterator

import arxiv
from dateutil.relativedelta import relativedelta

from config import (
    ARXIV_DELAY_SECONDS,
    ARXIV_INTER_CATEGORY_SLEEP,
    ARXIV_NUM_RETRIES,
    ARXIV_PAGE_SIZE,
    DEFAULT_CATEGORIES,
)
from db import (
    fetch_existing_arxiv_ids,
    get_connection,
    get_latest_published_at,
    get_state,
    init_db,
    prune_oldest_papers,
    set_state,
    upsert_papers,
    utc_now_iso,
)
from processor import extract_github_links, recompute_metric_scores, recompute_tfidf_keywords, sync_alias_keywords


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("ingest")

BACKFILL_CURSOR_KEY = "backfill_cursor_index"
BACKFILL_HASH_KEY = "backfill_plan_hash"
BACKFILL_ANCHOR_KEY = "backfill_anchor_end_utc"
LAST_DAILY_KEY = "last_daily_ingest_utc"
TRANSIENT_ERROR_PATTERNS = (
    "HTTP 429",
    "HTTP 500",
    "HTTP 502",
    "HTTP 503",
    "HTTP 504",
    "timed out",
    "timeout",
    "Connection reset",
    "Temporary failure",
)


def normalize_arxiv_id(raw: str) -> str:
    value = raw.strip()
    if re.search(r"v\d+$", value):
        return re.sub(r"v\d+$", "", value)
    return value


def format_arxiv_date(value: datetime) -> str:
    dt = value.astimezone(timezone.utc)
    return dt.strftime("%Y%m%d%H%M")


def iter_month_windows(start: datetime, end: datetime) -> Iterator[tuple[datetime, datetime]]:
    cursor = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    while cursor < end:
        next_cursor = cursor + relativedelta(months=1)
        yield max(start, cursor), min(end, next_cursor - timedelta(minutes=1))
        cursor = next_cursor


def build_backfill_windows(
    categories: list[str], years: int, now: datetime | None = None
) -> list[tuple[str, datetime, datetime]]:
    now = now or datetime.now(timezone.utc)
    start = now - relativedelta(years=years)
    windows: list[tuple[str, datetime, datetime]] = []
    for category in categories:
        for window_start, window_end in iter_month_windows(start, now):
            windows.append((category, window_start, window_end))
    return windows


def hash_backfill_plan(windows: list[tuple[str, datetime, datetime]]) -> str:
    payload = "|".join(
        f"{category}:{window_start.isoformat()}:{window_end.isoformat()}"
        for category, window_start, window_end in windows
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _build_client() -> arxiv.Client:
    return arxiv.Client(
        page_size=ARXIV_PAGE_SIZE,
        delay_seconds=ARXIV_DELAY_SECONDS,
        num_retries=ARXIV_NUM_RETRIES,
    )


def _to_record(result: arxiv.Result) -> dict[str, object]:
    arxiv_id = normalize_arxiv_id(result.get_short_id())
    title = " ".join(result.title.split()) if result.title else ""
    abstract = " ".join(result.summary.split()) if result.summary else ""
    github_links = extract_github_links(abstract)
    categories = list(result.categories or [])
    published_at = result.published
    updated_at = result.updated
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "authors_json": [author.name for author in result.authors],
        "published_at": published_at.astimezone(timezone.utc).replace(microsecond=0).isoformat(),
        "updated_at": updated_at.astimezone(timezone.utc).replace(microsecond=0).isoformat(),
        "primary_category": result.primary_category or "",
        "all_categories_json": categories,
        "doi": result.doi,
        "entry_url": result.entry_id,
        "pdf_url": result.pdf_url,
        "has_code": bool(github_links),
        "github_links_json": github_links,
        "ingested_at": utc_now_iso(),
    }


def fetch_window(
    client: arxiv.Client,
    category: str,
    start: datetime,
    end: datetime,
) -> list[dict[str, object]]:
    query = f"cat:{category} AND submittedDate:[{format_arxiv_date(start)} TO {format_arxiv_date(end)}]"
    search = arxiv.Search(
        query=query,
        max_results=None,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Ascending,
    )
    records = []
    for result in client.results(search):
        records.append(_to_record(result))
    return records


def _is_transient_error(exc: Exception) -> bool:
    message = str(exc)
    return any(pattern in message for pattern in TRANSIENT_ERROR_PATTERNS)


def _fetch_window_with_retry(
    client: arxiv.Client,
    category: str,
    start: datetime,
    end: datetime,
    max_attempts: int,
    retry_base_seconds: int,
    retry_max_seconds: int,
) -> list[dict[str, object]]:
    attempts = 0
    while True:
        attempts += 1
        try:
            return fetch_window(client, category, start, end)
        except Exception as exc:
            if not _is_transient_error(exc):
                raise
            if attempts >= max_attempts:
                raise
            cooldown = min(retry_base_seconds * (2 ** (attempts - 1)), retry_max_seconds)
            LOGGER.warning(
                "Transient API error on %s %s->%s (attempt %s/%s). Cooldown %ss before retry.",
                category,
                start.isoformat(),
                end.isoformat(),
                attempts,
                max_attempts,
                cooldown,
            )
            time.sleep(cooldown)


def run_init(
    conn,
    years: int,
    categories: list[str],
    window_max_attempts: int = 20,
    window_retry_base_seconds: int = 120,
    window_retry_max_seconds: int = 3600,
) -> int:
    cursor_idx = int(get_state(conn, BACKFILL_CURSOR_KEY, "0") or "0")
    anchor_raw = get_state(conn, BACKFILL_ANCHOR_KEY)
    if anchor_raw and cursor_idx > 0:
        try:
            now = datetime.fromisoformat(anchor_raw)
        except ValueError:
            now = datetime.now(timezone.utc).replace(microsecond=0)
            set_state(conn, BACKFILL_ANCHOR_KEY, now.isoformat())
        else:
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            now = now.astimezone(timezone.utc).replace(microsecond=0)
    else:
        now = datetime.now(timezone.utc).replace(microsecond=0)
        set_state(conn, BACKFILL_ANCHOR_KEY, now.isoformat())

    windows = build_backfill_windows(categories, years=years, now=now)
    if not windows:
        LOGGER.info("No backfill windows generated.")
        return 0

    plan_hash = hash_backfill_plan(windows)
    stored_hash = get_state(conn, BACKFILL_HASH_KEY)
    if stored_hash != plan_hash:
        cursor_idx = 0
        set_state(conn, BACKFILL_HASH_KEY, plan_hash)
        set_state(conn, BACKFILL_CURSOR_KEY, "0")
        set_state(conn, BACKFILL_ANCHOR_KEY, now.isoformat())

    total_written = 0
    client = _build_client()
    for idx in range(cursor_idx, len(windows)):
        category, start, end = windows[idx]
        LOGGER.info(
            "Backfill window %s/%s | %s | %s -> %s",
            idx + 1,
            len(windows),
            category,
            start.isoformat(),
            end.isoformat(),
        )
        try:
            records = _fetch_window_with_retry(
                client=client,
                category=category,
                start=start,
                end=end,
                max_attempts=window_max_attempts,
                retry_base_seconds=window_retry_base_seconds,
                retry_max_seconds=window_retry_max_seconds,
            )
        except Exception:
            LOGGER.exception("Failed to fetch window %s for %s", idx + 1, category)
            break
        count = upsert_papers(conn, records)
        total_written += count
        set_state(conn, BACKFILL_CURSOR_KEY, str(idx + 1))
        time.sleep(ARXIV_INTER_CATEGORY_SLEEP)

    if int(get_state(conn, BACKFILL_CURSOR_KEY, "0") or "0") >= len(windows):
        set_state(conn, "backfill_completed_at", utc_now_iso())
        set_state(conn, BACKFILL_ANCHOR_KEY, now.isoformat())
        LOGGER.info("Backfill completed. Recomputing keywords and scores.")
        recompute_tfidf_keywords(conn)
        sync_alias_keywords(conn)
        recompute_metric_scores(conn)
    else:
        LOGGER.info(
            "Backfill paused at cursor=%s/%s",
            get_state(conn, BACKFILL_CURSOR_KEY, "0"),
            len(windows),
        )
    return total_written


def run_daily(conn, window_hours: int, categories: list[str]) -> int:
    now = datetime.now(timezone.utc)
    since_raw = get_state(conn, LAST_DAILY_KEY)
    if since_raw:
        since = datetime.fromisoformat(since_raw)
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        since = since.astimezone(timezone.utc)
    else:
        since = now - timedelta(hours=window_hours)

    total_written = 0
    client = _build_client()
    for category in categories:
        LOGGER.info("Daily fetch | %s | %s -> %s", category, since.isoformat(), now.isoformat())
        try:
            records = _fetch_window_with_retry(
                client=client,
                category=category,
                start=since,
                end=now,
                max_attempts=10,
                retry_base_seconds=60,
                retry_max_seconds=1200,
            )
        except Exception:
            LOGGER.exception("Daily fetch failed for category %s", category)
            continue
        count = upsert_papers(conn, records)
        total_written += count
        time.sleep(ARXIV_INTER_CATEGORY_SLEEP)

    set_state(conn, LAST_DAILY_KEY, now.replace(microsecond=0).isoformat())
    if total_written > 0:
        recompute_tfidf_keywords(conn)
        sync_alias_keywords(conn)
        recompute_metric_scores(conn)
    LOGGER.info("Daily ingest wrote %s records.", total_written)
    return total_written


def run_daily_from_latest_with_prune(
    conn,
    categories: list[str],
    prune_equal_inserted: bool = True,
) -> dict[str, object]:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    latest_raw = get_latest_published_at(conn)
    if latest_raw:
        try:
            since = datetime.fromisoformat(latest_raw)
        except ValueError:
            since = now - timedelta(hours=24)
        else:
            if since.tzinfo is None:
                since = since.replace(tzinfo=timezone.utc)
            since = since.astimezone(timezone.utc).replace(microsecond=0)
    else:
        since = now - timedelta(hours=24)

    client = _build_client()
    failed_categories: list[str] = []
    dedup_records: dict[str, dict[str, object]] = {}

    for category in categories:
        LOGGER.info("Daily UI update fetch | %s | %s -> %s", category, since.isoformat(), now.isoformat())
        try:
            records = _fetch_window_with_retry(
                client=client,
                category=category,
                start=since,
                end=now,
                max_attempts=10,
                retry_base_seconds=60,
                retry_max_seconds=1200,
            )
        except Exception:
            LOGGER.exception("Daily UI update failed for category %s", category)
            failed_categories.append(category)
            continue
        for rec in records:
            dedup_records[str(rec["arxiv_id"])] = rec
        time.sleep(ARXIV_INTER_CATEGORY_SLEEP)

    records_to_upsert = list(dedup_records.values())
    existing_ids = fetch_existing_arxiv_ids(conn, dedup_records.keys())
    inserted_count = sum(1 for arxiv_id in dedup_records if arxiv_id not in existing_ids)
    inserted_arxiv_ids = [arxiv_id for arxiv_id in dedup_records if arxiv_id not in existing_ids]
    processed_count = upsert_papers(conn, records_to_upsert)

    deleted_count = 0
    if prune_equal_inserted and inserted_count > 0:
        deleted_count = prune_oldest_papers(conn, inserted_count)

    set_state(conn, LAST_DAILY_KEY, now.isoformat())
    return {
        "since": since.isoformat(),
        "until": now.isoformat(),
        "processed": processed_count,
        "inserted": inserted_count,
        "inserted_arxiv_ids": inserted_arxiv_ids,
        "deleted": deleted_count,
        "failed_categories": failed_categories,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArXiv Insight Tracker ingestion pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Backfill past N years by monthly chunks")
    init_parser.add_argument("--years", type=int, default=5, help="How many years to backfill")
    init_parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="arXiv categories to backfill",
    )
    init_parser.add_argument(
        "--window-max-attempts",
        type=int,
        default=20,
        help="Max retry attempts for a single month window on transient API errors",
    )
    init_parser.add_argument(
        "--window-retry-base-seconds",
        type=int,
        default=120,
        help="Base cooldown seconds for window retries (exponential backoff)",
    )
    init_parser.add_argument(
        "--window-retry-max-seconds",
        type=int,
        default=3600,
        help="Max cooldown seconds for window retries",
    )

    daily_parser = subparsers.add_parser("daily", help="Fetch recent increment")
    daily_parser.add_argument(
        "--window-hours",
        type=int,
        default=24,
        help="Fallback increment window when state is missing",
    )
    daily_parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="arXiv categories for increment pull",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conn = get_connection()
    init_db(conn)

    if args.command == "init":
        inserted = run_init(
            conn,
            years=args.years,
            categories=args.categories,
            window_max_attempts=args.window_max_attempts,
            window_retry_base_seconds=args.window_retry_base_seconds,
            window_retry_max_seconds=args.window_retry_max_seconds,
        )
    elif args.command == "daily":
        inserted = run_daily(conn, window_hours=args.window_hours, categories=args.categories)
    else:
        raise ValueError(f"Unknown command: {args.command}")

    print(json.dumps({"command": args.command, "inserted": inserted}, ensure_ascii=True))


if __name__ == "__main__":
    main()
