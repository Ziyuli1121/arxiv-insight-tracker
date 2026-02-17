from __future__ import annotations

import json
import re
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    DEFAULT_CROSS_DOMAIN_PREFIXES,
    FRONTIER_FRESHNESS_WEIGHT,
    FRONTIER_KEYWORD_WEIGHT,
    FRONTIER_OPEN_SOURCE_WEIGHT,
    KEYWORD_BURST_BASELINE_MONTHS,
    KEYWORD_BURST_RECENT_MONTHS,
    OPEN_SOURCE_HAS_CODE_WEIGHT,
    OPEN_SOURCE_STAR_WEIGHT,
    TFIDF_MAX_FEATURES,
    TFIDF_TOP_K,
)
from db import get_active_tracked_terms, replace_alias_keywords, replace_tfidf_keywords, upsert_metrics


GITHUB_PATTERN = re.compile(
    r"https?://(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)(?:/[^\s)\]]*)?",
    re.IGNORECASE,
)
BOOL_TOKEN_PATTERN = re.compile(r'"[^"]+"|\bAND\b|\bOR\b|\bNOT\b|[^\s]+', re.IGNORECASE)
SORT_SQL_MAP = {
    "newest": "p.published_at",
    "open_source": "COALESCE(m.open_source_score, 0)",
    "frontier": "COALESCE(m.frontier_score, 0)",
}


def extract_github_links(text: str) -> list[str]:
    if not text:
        return []
    unique = []
    seen = set()
    for owner, repo in GITHUB_PATTERN.findall(text):
        normalized = f"https://github.com/{owner}/{repo}".rstrip("/")
        normalized = normalized.rstrip(".,:;)")
        key = normalized.lower()
        if key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique


def parse_terms_input(raw: str) -> list[str]:
    if not raw:
        return []
    terms = [chunk.strip() for chunk in raw.split(",")]
    return [t for t in terms if t]


def _normalize_text(value: str) -> str:
    return " ".join((value or "").lower().split())


def _to_fts_phrase(term: str) -> str:
    normalized = " ".join((term or "").strip().split())
    if not normalized:
        return ""
    escaped = normalized.replace('"', '""')
    # Enforce exact token/phrase match in FTS.
    return f'"{escaped}"'


def build_combined_fts_query(
    boolean_query: str | None = None,
    exact_terms: Iterable[str] | None = None,
) -> str | None:
    bool_query = normalize_boolean_query(boolean_query or "")
    term_parts = [_to_fts_phrase(term) for term in (exact_terms or []) if term and term.strip()]
    term_parts = [part for part in term_parts if part]
    exact_query = " OR ".join(term_parts)

    if bool_query and exact_query:
        return f"({bool_query}) AND ({exact_query})"
    if bool_query:
        return bool_query
    if exact_query:
        return exact_query
    return None


def _tokenize_boolean_query(query: str) -> list[str]:
    return BOOL_TOKEN_PATTERN.findall(query or "")


def _normalize_boolean_tokens(tokens: list[str]) -> list[str]:
    normalized: list[str] = []
    previous_was_term = False
    for token in tokens:
        upper = token.upper()
        if upper in {"AND", "OR", "NOT"}:
            if upper == "NOT" and previous_was_term:
                normalized.append("AND")
            normalized.append(upper)
            previous_was_term = False
            continue
        term = token.strip()
        if term.startswith('"') and term.endswith('"') and len(term) >= 2:
            term = term[1:-1]
        if not term:
            continue
        if previous_was_term:
            normalized.append("AND")
        normalized.append(term)
        previous_was_term = True
    return normalized


def normalize_boolean_query(query: str) -> str:
    tokens = _normalize_boolean_tokens(_tokenize_boolean_query(query))
    if not tokens:
        return ""
    parts: list[str] = []
    for token in tokens:
        upper = token.upper()
        if upper in {"AND", "OR", "NOT"}:
            parts.append(upper)
            continue
        phrase = _to_fts_phrase(token)
        if phrase:
            parts.append(phrase)
    return " ".join(parts)


def _resolve_sort_column(sort_key: str | None) -> str:
    key = (sort_key or "newest").strip().lower()
    return SORT_SQL_MAP.get(key, SORT_SQL_MAP["newest"])


def _to_postfix(tokens: list[str]) -> list[str]:
    precedence = {"OR": 1, "AND": 2, "NOT": 3}
    output: list[str] = []
    stack: list[str] = []
    for token in tokens:
        upper = token.upper()
        if upper in precedence:
            while stack:
                top = stack[-1]
                if precedence.get(top, 0) >= precedence[upper]:
                    output.append(stack.pop())
                else:
                    break
            stack.append(upper)
        else:
            output.append(token)
    while stack:
        output.append(stack.pop())
    return output


def _evaluate_postfix_for_text(postfix: list[str], text: str) -> bool:
    stack: list[bool] = []
    for token in postfix:
        if token == "NOT":
            if not stack:
                raise ValueError("Invalid query around NOT")
            stack.append(not stack.pop())
            continue
        if token in {"AND", "OR"}:
            if len(stack) < 2:
                raise ValueError("Invalid boolean expression")
            right = stack.pop()
            left = stack.pop()
            stack.append(left and right if token == "AND" else left or right)
            continue
        stack.append(token.lower() in text)
    if len(stack) != 1:
        raise ValueError("Invalid boolean query")
    return stack[0]


def filter_by_boolean_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if df.empty or not query or not query.strip():
        return df
    tokens = _normalize_boolean_tokens(_tokenize_boolean_query(query))
    if not tokens:
        return df
    postfix = _to_postfix(tokens)
    if "_search_text" in df.columns:
        text_series = df["_search_text"].fillna("")
    else:
        title_series = df["title"].fillna("") if "title" in df.columns else pd.Series("", index=df.index)
        abstract_series = df["abstract"].fillna("") if "abstract" in df.columns else pd.Series("", index=df.index)
        text_series = title_series + " " + abstract_series
    try:
        mask = text_series.apply(lambda txt: _evaluate_postfix_for_text(postfix, _normalize_text(str(txt))))
    except ValueError:
        return df
    return df[mask].copy()


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            pass
        return [value]
    return [str(value)]


def get_tracked_terms_map(conn: sqlite3.Connection) -> dict[str, list[str]]:
    rows = get_active_tracked_terms(conn)
    mapping: dict[str, list[str]] = {}
    for row in rows:
        aliases = _as_list(row["aliases_json"])
        mapping[row["term"]] = aliases
    return mapping


def fts_prefilter_ids(conn: sqlite3.Connection, query: str, limit: int = 50000) -> set[str] | None:
    if not query or not query.strip():
        return None
    candidate = query.strip()
    try:
        rows = conn.execute(
            """
            SELECT arxiv_id
            FROM papers_fts
            WHERE papers_fts MATCH ?
            LIMIT ?
            """,
            (candidate, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return None
    return {row["arxiv_id"] for row in rows}


def load_papers_dataframe(
    conn: sqlite3.Connection,
    start_date: date | datetime | None = None,
    end_date: date | datetime | None = None,
    categories: Iterable[str] | None = None,
    fts_query: str | None = None,
    has_code_only: bool = False,
    sort_key: str = "newest",
    max_rows: int | None = None,
) -> pd.DataFrame:
    clauses = []
    params: list[object] = []

    if start_date is not None:
        clauses.append("p.published_at >= ?")
        params.append(_to_iso_day_start(start_date))
    if end_date is not None:
        clauses.append("p.published_at <= ?")
        params.append(_to_iso_day_end(end_date))

    categories_list = list(categories or [])
    if categories_list:
        placeholders = ",".join("?" for _ in categories_list)
        clauses.append(f"p.primary_category IN ({placeholders})")
        params.extend(categories_list)
    if fts_query and fts_query.strip():
        clauses.append("p.arxiv_id IN (SELECT arxiv_id FROM papers_fts WHERE papers_fts MATCH ?)")
        params.append(fts_query.strip())
    if has_code_only:
        clauses.append("p.has_code = 1")

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    order_column = _resolve_sort_column(sort_key)
    order_sql = f"ORDER BY {order_column} DESC, p.published_at DESC"
    limit_sql = ""
    if max_rows is not None and int(max_rows) > 0:
        limit_sql = "LIMIT ?"
        params.append(int(max_rows))
    sql = f"""
    SELECT
        p.arxiv_id,
        p.title,
        p.abstract,
        p.authors_json,
        p.published_at,
        p.updated_at,
        p.primary_category,
        p.all_categories_json,
        p.doi,
        p.entry_url,
        p.pdf_url,
        p.has_code,
        p.github_links_json,
        m.frontier_score,
        m.github_stars_max,
        m.open_source_score,
        m.keyword_burst,
        m.freshness
    FROM papers p
    LEFT JOIN paper_metrics m ON m.arxiv_id = p.arxiv_id
    {where}
    {order_sql}
    {limit_sql}
    """
    df = pd.read_sql_query(sql, conn, params=params)
    if df.empty:
        return df

    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True, errors="coerce")
    for col in [
        "frontier_score",
        "github_stars_max",
        "open_source_score",
        "keyword_burst",
        "freshness",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["has_code"] = df["has_code"].fillna(0).astype(int)
    return df


def query_papers(
    conn: sqlite3.Connection,
    start_date: date | datetime | None = None,
    end_date: date | datetime | None = None,
    categories: Iterable[str] | None = None,
    boolean_query: str | None = None,
    exact_terms: Iterable[str] | None = None,
    has_code_only: bool = False,
    sort_key: str = "newest",
    max_rows: int | None = None,
) -> pd.DataFrame:
    fts_query = build_combined_fts_query(boolean_query=boolean_query, exact_terms=exact_terms)
    try:
        return load_papers_dataframe(
            conn,
            start_date=start_date,
            end_date=end_date,
            categories=categories,
            fts_query=fts_query,
            has_code_only=has_code_only,
            sort_key=sort_key,
            max_rows=max_rows,
        )
    except sqlite3.OperationalError:
        return pd.DataFrame()


def _to_period_series(series: pd.Series, granularity: str) -> pd.Series:
    if granularity == "week":
        return series.dt.to_period("W").dt.start_time
    if granularity == "day":
        return series.dt.to_period("D").dt.start_time
    return series.dt.to_period("M").dt.start_time


def _period_freq(granularity: str) -> str:
    if granularity == "week":
        return "W-MON"
    if granularity == "day":
        return "D"
    return "MS"


def _period_sql_expr(granularity: str, column: str = "p.published_at") -> str:
    if granularity == "week":
        # Monday-start week bucket.
        return (
            f"date({column}, '-' || ((CAST(strftime('%w', {column}) AS integer) + 6) % 7) || ' days')"
        )
    if granularity == "day":
        return f"date({column})"
    return f"strftime('%Y-%m-01', {column})"


def _period_floor(value: date | datetime, granularity: str) -> datetime:
    if isinstance(value, datetime):
        dt = value.astimezone(timezone.utc)
    else:
        dt = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if granularity == "day":
        return dt
    if granularity == "week":
        return dt - timedelta(days=dt.weekday())
    return dt.replace(day=1)


def _build_base_clauses_and_params(
    start_date: date | datetime | None = None,
    end_date: date | datetime | None = None,
    categories: Iterable[str] | None = None,
    fts_query: str | None = None,
    has_code_only: bool = False,
) -> tuple[list[str], list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if start_date is not None:
        clauses.append("p.published_at >= ?")
        params.append(_to_iso_day_start(start_date))
    if end_date is not None:
        clauses.append("p.published_at <= ?")
        params.append(_to_iso_day_end(end_date))

    categories_list = list(categories or [])
    if categories_list:
        placeholders = ",".join("?" for _ in categories_list)
        clauses.append(f"p.primary_category IN ({placeholders})")
        params.extend(categories_list)

    if fts_query and fts_query.strip():
        clauses.append("p.arxiv_id IN (SELECT arxiv_id FROM papers_fts WHERE papers_fts MATCH ?)")
        params.append(fts_query.strip())
    if has_code_only:
        clauses.append("p.has_code = 1")
    return clauses, params


def build_macro_trends(
    conn: sqlite3.Connection,
    terms: Iterable[str],
    start_date: date | datetime | None = None,
    end_date: date | datetime | None = None,
    categories: Iterable[str] | None = None,
    boolean_query: str | None = None,
    exact_terms: Iterable[str] | None = None,
    granularity: str = "month",
    aliases_map: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    aliases_map = aliases_map or {}
    terms_list = [t for t in terms if t]
    if not terms_list:
        return pd.DataFrame(columns=["period", "term", "count"])

    base_fts = build_combined_fts_query(boolean_query=boolean_query, exact_terms=exact_terms)
    base_clauses, base_params = _build_base_clauses_and_params(
        start_date=start_date,
        end_date=end_date,
        categories=categories,
        fts_query=base_fts,
    )
    period_expr = _period_sql_expr(granularity)

    where_base = f"WHERE {' AND '.join(base_clauses)}" if base_clauses else ""
    if start_date is not None and end_date is not None:
        period_start = _period_floor(start_date, granularity)
        period_end = _period_floor(end_date, granularity)
    else:
        bounds_sql = f"SELECT MIN({period_expr}) AS min_period, MAX({period_expr}) AS max_period FROM papers p {where_base}"
        try:
            bounds = conn.execute(bounds_sql, base_params).fetchone()
        except sqlite3.OperationalError:
            return pd.DataFrame(columns=["period", "term", "count"])
        if bounds is None or bounds["min_period"] is None or bounds["max_period"] is None:
            return pd.DataFrame(columns=["period", "term", "count"])
        period_start = pd.to_datetime(bounds["min_period"], utc=True).to_pydatetime()
        period_end = pd.to_datetime(bounds["max_period"], utc=True).to_pydatetime()

    periods = pd.date_range(period_start, period_end, freq=_period_freq(granularity))
    if len(periods) == 0:
        return pd.DataFrame(columns=["period", "term", "count"])

    frames = []
    for term in terms_list:
        variants = [term] + aliases_map.get(term, [])
        term_fts = build_combined_fts_query(exact_terms=variants)
        if base_fts and term_fts:
            final_fts = f"({base_fts}) AND ({term_fts})"
        else:
            final_fts = term_fts or base_fts

        clauses, params = _build_base_clauses_and_params(
            start_date=start_date,
            end_date=end_date,
            categories=categories,
            fts_query=final_fts,
        )
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
        SELECT {period_expr} AS period, COUNT(*) AS count
        FROM papers p
        {where}
        GROUP BY period
        ORDER BY period
        """
        try:
            grouped = pd.read_sql_query(sql, conn, params=params)
        except sqlite3.OperationalError:
            grouped = pd.DataFrame(columns=["period", "count"])
        if grouped.empty:
            counts = pd.Series(dtype="int64")
        else:
            grouped["period"] = pd.to_datetime(grouped["period"], utc=True, errors="coerce")
            counts = grouped.set_index("period")["count"]
        frame = pd.DataFrame({"period": periods})
        frame["count"] = frame["period"].map(counts).fillna(0).astype(int)
        frame["term"] = term
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_streamgraph_data(
    conn: sqlite3.Connection,
    start_date: date | datetime | None = None,
    end_date: date | datetime | None = None,
    categories: Iterable[str] | None = None,
    boolean_query: str | None = None,
    exact_terms: Iterable[str] | None = None,
    granularity: str = "month",
) -> pd.DataFrame:
    base_fts = build_combined_fts_query(boolean_query=boolean_query, exact_terms=exact_terms)
    clauses, params = _build_base_clauses_and_params(
        start_date=start_date,
        end_date=end_date,
        categories=categories,
        fts_query=base_fts,
    )
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    period_expr = _period_sql_expr(granularity)
    sql = f"""
    SELECT
        {period_expr} AS period,
        p.primary_category AS primary_category,
        COUNT(*) AS count
    FROM papers p
    {where}
    GROUP BY period, primary_category
    ORDER BY period, primary_category
    """
    try:
        grouped = pd.read_sql_query(sql, conn, params=params)
    except sqlite3.OperationalError:
        grouped = pd.DataFrame(columns=["period", "primary_category", "count"])
    if grouped.empty:
        return pd.DataFrame(columns=["period", "primary_category", "count"])
    grouped["period"] = pd.to_datetime(grouped["period"], utc=True, errors="coerce")
    grouped["count"] = pd.to_numeric(grouped["count"], errors="coerce").fillna(0).astype(int)
    return grouped


def build_ai4s_df(
    df: pd.DataFrame,
    cross_prefixes: Iterable[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df
    prefixes = [prefix.lower() for prefix in (cross_prefixes or DEFAULT_CROSS_DOMAIN_PREFIXES)]
    if "categories" in df.columns:
        categories_col = df["categories"]
    elif "all_categories_json" in df.columns:
        categories_col = df["all_categories_json"].apply(_as_list)
    else:
        categories_col = pd.Series([[] for _ in range(len(df))], index=df.index)

    def is_ai4s(categories: list[str]) -> bool:
        cat_lower = [c.lower() for c in categories]
        has_cs = any(c.startswith("cs.") for c in cat_lower)
        has_cross = any(any(c.startswith(prefix) for prefix in prefixes) for c in cat_lower)
        return has_cs and has_cross

    mask = categories_col.apply(is_ai4s)
    return df[mask].copy()


def build_ai4s_domain_trend(
    df: pd.DataFrame,
    cross_prefixes: Iterable[str] | None = None,
    granularity: str = "month",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["period", "domain", "count"])
    prefixes = [prefix.lower() for prefix in (cross_prefixes or DEFAULT_CROSS_DOMAIN_PREFIXES)]
    if "categories" in df.columns:
        categories_col = df["categories"]
    elif "all_categories_json" in df.columns:
        categories_col = df["all_categories_json"].apply(_as_list)
    else:
        categories_col = pd.Series([[] for _ in range(len(df))], index=df.index)

    rows: list[dict[str, object]] = []
    for idx, row in df.iterrows():
        cats = [c.lower() for c in categories_col.loc[idx]]
        for prefix in prefixes:
            if any(cat.startswith(prefix) for cat in cats):
                rows.append(
                    {
                        "arxiv_id": row["arxiv_id"],
                        "domain": prefix,
                        "published_at": row["published_at"],
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["period", "domain", "count"])
    tmp = pd.DataFrame(rows).drop_duplicates(subset=["arxiv_id", "domain"])
    tmp["period"] = _to_period_series(pd.to_datetime(tmp["published_at"], utc=True), granularity)
    return tmp.groupby(["period", "domain"]).size().reset_index(name="count")


def compute_keyword_momentum(
    conn: sqlite3.Connection,
    arxiv_ids: Iterable[str] | None = None,
    lookback_months: int = KEYWORD_BURST_RECENT_MONTHS,
    baseline_months: int = KEYWORD_BURST_BASELINE_MONTHS,
    limit: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = list(arxiv_ids or [])
    if not ids:
        kw_df = pd.read_sql_query(
            """
            SELECT pk.keyword, p.published_at
            FROM paper_keywords pk
            JOIN papers p ON p.arxiv_id = pk.arxiv_id
            WHERE pk.source = 'tfidf'
            """,
            conn,
        )
    else:
        chunks: list[pd.DataFrame] = []
        chunk_size = 800
        for start in range(0, len(ids), chunk_size):
            chunk = ids[start : start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            sql = f"""
            SELECT pk.keyword, p.published_at
            FROM paper_keywords pk
            JOIN papers p ON p.arxiv_id = pk.arxiv_id
            WHERE pk.source = 'tfidf' AND p.arxiv_id IN ({placeholders})
            """
            part = pd.read_sql_query(sql, conn, params=chunk)
            if not part.empty:
                chunks.append(part)
        kw_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["keyword", "published_at"])
    if kw_df.empty:
        empty = pd.DataFrame(columns=["keyword", "recent_rate", "baseline_rate", "delta", "ratio"])
        return empty, empty

    kw_df["published_at"] = pd.to_datetime(kw_df["published_at"], utc=True)
    now = kw_df["published_at"].max()
    recent_start = now - relativedelta(months=lookback_months)
    baseline_start = recent_start - relativedelta(months=baseline_months)

    recent = kw_df[kw_df["published_at"] >= recent_start].groupby("keyword").size().rename("recent_count")
    baseline = (
        kw_df[(kw_df["published_at"] >= baseline_start) & (kw_df["published_at"] < recent_start)]
        .groupby("keyword")
        .size()
        .rename("baseline_count")
    )
    stats = pd.concat([recent, baseline], axis=1).fillna(0.0)
    stats["recent_rate"] = stats["recent_count"] / max(lookback_months, 1)
    stats["baseline_rate"] = stats["baseline_count"] / max(baseline_months, 1)
    stats["delta"] = stats["recent_rate"] - stats["baseline_rate"]
    stats["ratio"] = stats["recent_rate"] / (stats["baseline_rate"] + 1e-6)
    stats = stats.reset_index()
    rising = stats.sort_values("delta", ascending=False).head(limit)
    declining = stats.sort_values("delta", ascending=True).head(limit)
    return rising, declining


def recompute_tfidf_keywords(
    conn: sqlite3.Connection,
    max_features: int = TFIDF_MAX_FEATURES,
    top_k: int = TFIDF_TOP_K,
) -> int:
    df = pd.read_sql_query("SELECT arxiv_id, title, abstract FROM papers", conn)
    if df.empty:
        replace_tfidf_keywords(conn, [])
        return 0

    corpus = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
    )
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        replace_tfidf_keywords(conn, [])
        return 0

    names = np.array(vectorizer.get_feature_names_out())
    rows: list[tuple[str, str, float, str]] = []
    for i, arxiv_id in enumerate(df["arxiv_id"].tolist()):
        row = matrix.getrow(i)
        if row.nnz == 0:
            continue
        order = np.argsort(row.data)[::-1][:top_k]
        indices = row.indices[order]
        scores = row.data[order]
        for idx, score in zip(indices, scores):
            rows.append((arxiv_id, str(names[idx]), float(score), "tfidf"))
    replace_tfidf_keywords(conn, rows)
    return len(rows)


def sync_alias_keywords(conn: sqlite3.Connection) -> int:
    terms = get_active_tracked_terms(conn)
    if not terms:
        replace_alias_keywords(conn, [])
        return 0

    rows: list[tuple[str, str, float, str]] = []
    for term_row in terms:
        term = str(term_row["term"])
        aliases = _as_list(term_row["aliases_json"])
        variants = [term] + aliases
        fts_query = build_combined_fts_query(exact_terms=variants)
        if not fts_query:
            continue
        try:
            matched = conn.execute(
                "SELECT arxiv_id FROM papers_fts WHERE papers_fts MATCH ?",
                (fts_query,),
            ).fetchall()
        except sqlite3.OperationalError:
            continue
        for row in matched:
            rows.append((row["arxiv_id"], term, 1.0, "alias"))
    replace_alias_keywords(conn, rows)
    return len(rows)


def recompute_metric_scores(conn: sqlite3.Connection) -> int:
    sql = """
    SELECT
        p.arxiv_id,
        p.published_at,
        p.has_code,
        COALESCE(m.github_stars_max, 0) AS github_stars_max,
        m.metrics_updated_at
    FROM papers p
    LEFT JOIN paper_metrics m ON m.arxiv_id = p.arxiv_id
    """
    df = pd.read_sql_query(sql, conn)
    if df.empty:
        return 0
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    now = pd.Timestamp(datetime.now(timezone.utc))
    age_days = (now - df["published_at"]).dt.days.fillna(365.0).clip(lower=0.0)
    df["github_stars_max"] = pd.to_numeric(df["github_stars_max"], errors="coerce").fillna(0.0)
    df["has_code"] = pd.to_numeric(df["has_code"], errors="coerce").fillna(0).astype(int)

    log_stars = np.log1p(df["github_stars_max"].fillna(0.0))
    if np.isclose(log_stars.max(), log_stars.min()):
        normalized_log_stars = pd.Series(0.0, index=df.index)
    else:
        normalized_log_stars = (log_stars - log_stars.min()) / (log_stars.max() - log_stars.min())
    df["open_source_score"] = (
        OPEN_SOURCE_HAS_CODE_WEIGHT * df["has_code"] + OPEN_SOURCE_STAR_WEIGHT * normalized_log_stars
    )

    keyword_burst_map = _compute_keyword_burst_by_paper(conn, reference_now=now.to_pydatetime())
    df["keyword_burst"] = df["arxiv_id"].map(keyword_burst_map).fillna(0.0)
    df["freshness"] = np.exp(-age_days / 365.0)

    df["frontier_score"] = (
        FRONTIER_KEYWORD_WEIGHT * df["keyword_burst"]
        + FRONTIER_OPEN_SOURCE_WEIGHT * df["open_source_score"]
        + FRONTIER_FRESHNESS_WEIGHT * df["freshness"]
    )

    rows = []
    for _, row in df.iterrows():
        metrics_updated_at = row.get("metrics_updated_at")
        if pd.isna(metrics_updated_at):
            metrics_updated_at = None
        rows.append(
            {
                "arxiv_id": row["arxiv_id"],
                "citation_count": 0,
                "citations_12m": 0,
                "citation_velocity": 0.0,
                "impact_score": 0.0,
                "frontier_score": float(row["frontier_score"]),
                "github_stars_max": int(row["github_stars_max"]),
                "open_source_score": float(row["open_source_score"]),
                "keyword_burst": float(row["keyword_burst"]),
                "freshness": float(row["freshness"]),
                "openalex_id": None,
                "metrics_updated_at": metrics_updated_at,
            }
        )
    upsert_metrics(conn, rows)
    return len(rows)


def _compute_keyword_burst_by_paper(
    conn: sqlite3.Connection,
    reference_now: datetime | None = None,
) -> dict[str, float]:
    sql = """
    SELECT pk.arxiv_id, pk.keyword, p.published_at
    FROM paper_keywords pk
    JOIN papers p ON p.arxiv_id = pk.arxiv_id
    WHERE pk.source = 'tfidf'
    """
    kw_df = pd.read_sql_query(sql, conn)
    if kw_df.empty:
        return {}
    kw_df["published_at"] = pd.to_datetime(kw_df["published_at"], utc=True, errors="coerce")
    ref = pd.Timestamp(reference_now or datetime.now(timezone.utc))
    recent_start = ref - relativedelta(months=KEYWORD_BURST_RECENT_MONTHS)
    baseline_start = recent_start - relativedelta(months=KEYWORD_BURST_BASELINE_MONTHS)

    recent_counts = kw_df[kw_df["published_at"] >= recent_start].groupby("keyword").size().rename("recent")
    baseline_counts = (
        kw_df[(kw_df["published_at"] >= baseline_start) & (kw_df["published_at"] < recent_start)]
        .groupby("keyword")
        .size()
        .rename("baseline")
    )
    stats = pd.concat([recent_counts, baseline_counts], axis=1).fillna(0.0)
    stats["recent_rate"] = stats["recent"] / max(KEYWORD_BURST_RECENT_MONTHS, 1)
    stats["baseline_rate"] = stats["baseline"] / max(KEYWORD_BURST_BASELINE_MONTHS, 1)
    stats["burst_raw"] = stats["recent_rate"] / (stats["baseline_rate"] + 1e-6)
    stats["burst_log"] = np.log1p(stats["burst_raw"])
    if np.isclose(stats["burst_log"].max(), stats["burst_log"].min()):
        stats["burst_norm"] = 0.0
    else:
        stats["burst_norm"] = (stats["burst_log"] - stats["burst_log"].min()) / (
            stats["burst_log"].max() - stats["burst_log"].min()
        )
    keyword_map = stats["burst_norm"].to_dict()
    kw_df["keyword_score"] = kw_df["keyword"].map(keyword_map).fillna(0.0)
    paper_scores = kw_df.groupby("arxiv_id")["keyword_score"].max().to_dict()
    return {str(k): float(v) for k, v in paper_scores.items()}


def _to_iso_day_start(value: date | datetime) -> str:
    if isinstance(value, datetime):
        dt = value.astimezone(timezone.utc).replace(microsecond=0)
        return dt.isoformat()
    else:
        dt = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


def _to_iso_day_end(value: date | datetime) -> str:
    if isinstance(value, datetime):
        dt = value.astimezone(timezone.utc).replace(microsecond=0)
        return dt.isoformat()
    else:
        dt = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    dt = dt.replace(hour=23, minute=59, second=59, microsecond=0)
    return dt.isoformat()


def latest_24h(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    recent = df[df["published_at"] >= cutoff].copy()
    if not recent.empty:
        return recent
    # Fallback for stale local snapshots: show the latest 24h slice within the dataset.
    latest_ts = df["published_at"].max()
    if pd.isna(latest_ts):
        return recent
    fallback_cutoff = latest_ts - timedelta(hours=24)
    return df[df["published_at"] >= fallback_cutoff].copy()


def count_papers(
    conn: sqlite3.Connection,
    start_date: date | datetime | None = None,
    end_date: date | datetime | None = None,
    categories: Iterable[str] | None = None,
    boolean_query: str | None = None,
    exact_terms: Iterable[str] | None = None,
    has_code_only: bool = False,
) -> int:
    fts_query = build_combined_fts_query(boolean_query=boolean_query, exact_terms=exact_terms)
    clauses, params = _build_base_clauses_and_params(
        start_date=start_date,
        end_date=end_date,
        categories=categories,
        fts_query=fts_query,
        has_code_only=has_code_only,
    )
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT COUNT(*) FROM papers p {where}"
    try:
        row = conn.execute(sql, params).fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row[0] if row else 0)
