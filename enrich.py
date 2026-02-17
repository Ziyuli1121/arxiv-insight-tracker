from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any
from urllib.parse import quote

import requests

from config import (
    ENRICH_HTTP_MAX_RETRIES,
    GITHUB_API_BASE_URL,
    GITHUB_DELAY_SECONDS,
    GITHUB_TOKEN,
    REQUEST_TIMEOUT_SECONDS,
)
from db import (
    get_connection,
    get_papers_for_enrichment,
    get_papers_for_enrichment_by_ids,
    init_db,
    set_state,
    upsert_metrics,
    utc_now_iso,
)
from processor import recompute_metric_scores


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("enrich")


def _request_json(url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any | None:
    clean_params = {k: v for k, v in (params or {}).items() if v is not None}
    for attempt in range(ENRICH_HTTP_MAX_RETRIES):
        try:
            response = requests.get(
                url,
                params=clean_params or None,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code == 200:
                return response.json()
            if response.status_code in {429, 500, 502, 503, 504}:
                wait = 2**attempt
                time.sleep(wait)
                continue
            return None
        except requests.RequestException:
            wait = 2**attempt
            time.sleep(wait)
    return None


def _parse_repo(url: str) -> tuple[str, str] | None:
    marker = "github.com/"
    if marker not in (url or "").lower():
        return None
    try:
        tail = url.split("github.com/", 1)[1]
    except IndexError:
        return None
    parts = [p for p in tail.split("/") if p]
    if len(parts) < 2:
        return None
    owner, repo = parts[0], parts[1]
    repo = repo.strip()
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not owner or not repo:
        return None
    return owner, repo


def _get_github_stars(repo_url: str, cache: dict[str, int]) -> int | None:
    parsed = _parse_repo(repo_url)
    if parsed is None:
        return None
    owner, repo = parsed
    key = f"{owner}/{repo}".lower()
    if key in cache:
        return cache[key]
    if not GITHUB_TOKEN:
        return None
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = _request_json(f"{GITHUB_API_BASE_URL}/repos/{quote(owner)}/{quote(repo)}", headers=headers)
    time.sleep(GITHUB_DELAY_SECONDS)
    if not payload:
        return None
    stars = int(payload.get("stargazers_count", 0) or 0)
    cache[key] = stars
    return stars


def _safe_json_loads(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(v) for v in parsed]


def _run_sync_for_rows(conn, papers: list, progress_every: int = 50, recompute: bool = True) -> int:
    if not GITHUB_TOKEN:
        LOGGER.info("GITHUB_TOKEN is not set. Skipping GitHub stars enrichment.")
        return 0

    if not papers:
        LOGGER.info("No papers pending enrichment.")
        return 0

    now_iso = utc_now_iso()
    rows = []
    github_cache: dict[str, int] = {}
    with_stars_count = 0
    unresolved_count = 0
    total = len(papers)
    progress_every = max(1, int(progress_every))
    LOGGER.info("Starting enrichment batch: %s papers", total)

    for idx, paper in enumerate(papers, start=1):
        stars = []
        for link in _safe_json_loads(paper["github_links_json"]):
            star_count = _get_github_stars(link, github_cache)
            if star_count is not None:
                stars.append(star_count)
        github_stars_max = max(stars) if stars else 0
        if github_stars_max > 0:
            with_stars_count += 1
        else:
            unresolved_count += 1

        rows.append(
            {
                "arxiv_id": paper["arxiv_id"],
                "github_stars_max": github_stars_max,
                "open_source_score": 0.0,
                "keyword_burst": 0.0,
                "freshness": 0.0,
                "metrics_updated_at": now_iso,
            }
        )

        if idx % progress_every == 0 or idx == total:
            LOGGER.info(
                "Progress %s/%s | with_stars=%s | zero_or_unknown=%s",
                idx,
                total,
                with_stars_count,
                unresolved_count,
            )

    upsert_metrics(conn, rows)
    if recompute:
        recompute_metric_scores(conn)
    else:
        LOGGER.info("Skipped score recomputation for this batch (--no-recompute).")
    set_state(conn, "last_enrich_utc", now_iso)
    LOGGER.info("GitHub stars >0 for %s papers; zero/unknown for %s papers.", with_stars_count, unresolved_count)
    LOGGER.info("Enriched %s papers.", len(rows))
    return len(rows)


def run_sync(limit: int, progress_every: int = 50, recompute: bool = True) -> int:
    conn = get_connection()
    init_db(conn)
    papers = get_papers_for_enrichment(conn, limit=limit)
    return _run_sync_for_rows(conn, papers, progress_every=progress_every, recompute=recompute)


def run_sync_for_arxiv_ids(
    arxiv_ids: list[str],
    progress_every: int = 50,
    recompute: bool = False,
    conn=None,
) -> int:
    local_conn = conn
    if local_conn is None:
        local_conn = get_connection()
        init_db(local_conn)
    papers = get_papers_for_enrichment_by_ids(local_conn, arxiv_ids=arxiv_ids)
    return _run_sync_for_rows(local_conn, papers, progress_every=progress_every, recompute=recompute)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArXiv Insight Tracker enrichment pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    sync_parser = subparsers.add_parser("sync", help="Sync GitHub stars and refresh ranking scores")
    sync_parser.add_argument("--limit", type=int, default=300, help="Maximum papers to enrich in one run")
    sync_parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Log progress every N papers during a sync batch",
    )
    sync_parser.add_argument(
        "--no-recompute",
        action="store_true",
        help="Skip global metric score recomputation for this sync batch (faster).",
    )
    subparsers.add_parser("recompute", help="Recompute global metric scores for all papers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "sync":
        count = run_sync(limit=args.limit, progress_every=args.progress_every, recompute=not args.no_recompute)
    elif args.command == "recompute":
        conn = get_connection()
        init_db(conn)
        count = recompute_metric_scores(conn)
    else:
        raise ValueError(f"Unsupported command {args.command}")
    print(json.dumps({"command": args.command, "enriched": count}, ensure_ascii=True))


if __name__ == "__main__":
    main()
