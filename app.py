from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import streamlit as st
from dateutil.relativedelta import relativedelta

from db import get_connection, get_latest_published_at, init_db
from enrich import run_sync_for_arxiv_ids
from ingest import run_daily_from_latest_with_prune
from processor import (
    build_macro_trends,
    build_streamgraph_data,
    count_papers,
    compute_keyword_momentum,
    get_tracked_terms_map,
    parse_terms_input,
    query_papers,
)

@st.cache_resource
def get_conn():
    conn = get_connection()
    init_db(conn)
    return conn


def format_authors(authors: list[str], cap: int = 4) -> str:
    if not authors:
        return "Unknown authors"
    if len(authors) <= cap:
        return ", ".join(authors)
    return ", ".join(authors[:cap]) + ", et al."


def parse_json_list(value: object) -> list[str]:
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
            return []
    return []


def render_macro_trends(
    conn,
    start_date,
    end_date,
    selected_categories: list[str],
    bool_query: str,
    exact_filter_terms: list[str],
    trend_terms: list[str],
    granularity: str,
) -> None:
    st.subheader("Macro Trends")

    aliases_map = get_tracked_terms_map(conn)

    with st.spinner("Computing trend series..."):
        trend_df = build_macro_trends(
            conn,
            terms=trend_terms,
            start_date=start_date,
            end_date=end_date,
            categories=selected_categories,
            boolean_query=bool_query,
            exact_terms=exact_filter_terms,
            granularity=granularity,
            aliases_map=aliases_map,
        )
    if trend_df.empty:
        st.warning("No trend series available for the current keywords.")
    else:
        if int(trend_df["period"].nunique()) <= 1:
            st.info("Only one time bucket after aggregation. Expand date range or switch granularity to `week`/`day`.")
        fig_line = px.line(
            trend_df,
            x="period",
            y="count",
            color="term",
            markers=True,
            title="Keyword Publication Trends",
        )
        st.plotly_chart(fig_line, use_container_width=True)

    stream_df = build_streamgraph_data(
        conn,
        start_date=start_date,
        end_date=end_date,
        categories=selected_categories,
        boolean_query=bool_query,
        exact_terms=exact_filter_terms,
        granularity=granularity,
    )
    if not stream_df.empty:
        fig_area = px.area(
            stream_df,
            x="period",
            y="count",
            color="primary_category",
            title="Category Share Streamgraph",
        )
        st.plotly_chart(fig_area, use_container_width=True)

    if st.checkbox("Compute Keyword Momentum (slower)", value=False):
        with st.spinner("Computing keyword momentum..."):
            rising, declining = compute_keyword_momentum(conn, limit=10)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top Rising Keywords**")
            st.dataframe(rising, use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**Top Declining Keywords**")
            st.dataframe(declining, use_container_width=True, hide_index=True)


def render_daily_brief(df: pd.DataFrame) -> None:
    st.subheader("Daily Brief (DB Latest 24h Window)")
    daily_df = df.copy()
    limit = int(st.session_state.get("daily_cards_limit", 50))

    total = len(daily_df)
    code_count = int(daily_df["has_code"].sum()) if not daily_df.empty else 0
    avg_frontier = float(daily_df["frontier_score"].mean()) if not daily_df.empty else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Papers (24h)", total)
    c2.metric("With Code", code_count)
    c3.metric(
        "Avg Frontier",
        f"{avg_frontier:.3f}",
        help="Frontier = 0.55*keyword_burst + 0.30*open_source + 0.15*freshness.",
    )

    if daily_df.empty:
        st.info("No papers found for the current filters.")
        return

    for _, row in daily_df.head(limit).iterrows():
        title = row["title"]
        url = row["entry_url"]
        pub = row["published_at"].strftime("%Y-%m-%d %H:%M UTC")
        authors = format_authors(parse_json_list(row["authors_json"]))
        github_links = parse_json_list(row["github_links_json"])
        metrics = (
            f"OpenSource: {row['open_source_score']:.3f} | "
            f"Frontier: {row['frontier_score']:.3f}"
        )
        st.markdown(f"### [{title}]({url})")
        st.caption(f"{authors} | {pub} | {row['primary_category']}")
        with st.expander("Abstract", expanded=False):
            abstract = str(row.get("abstract", "") or "").strip()
            st.write(abstract if abstract else "(No abstract available)")
        st.markdown(metrics)
        if row["has_code"] and github_links:
            links = " ".join([f"[GitHub]({link})" for link in github_links])
            stars = int(row["github_stars_max"]) if pd.notna(row["github_stars_max"]) else 0
            st.markdown(f"`Code` {links} | Stars(max): {stars}")
        st.divider()


def main() -> None:
    st.set_page_config(page_title="ArXiv Insight Tracker", layout="wide")
    st.title("ArXiv Insight Tracker")
    st.caption("5-year AI trend intelligence with daily incremental refresh.")

    conn = get_conn()
    categories = ["cs.AI", "cs.CL", "cs.CV", "cs.LG"]

    tracked_map = get_tracked_terms_map(conn)
    default_terms = ", ".join(tracked_map.keys()) if tracked_map else "Diffusion, LLM, Agent"

    now = datetime.now(timezone.utc)
    default_start = (now - relativedelta(months=12)).date()
    default_end = now.date()

    with st.sidebar:
        st.header("Filters")
        view = st.radio("View", options=["Macro Trends", "Daily Brief"], index=0)
        st.caption("Default window is 12 months for faster loading. Keyword matching is exact (title/abstract token/phrase).")
        update_pressed = False
        with st.form("filters_form"):
            date_range = (default_start, default_end)

            st.markdown("**Common Filters**")
            if view == "Macro Trends":
                date_range = st.date_input(
                    "Date Range (UTC)",
                    value=(default_start, default_end),
                    min_value=(now - relativedelta(years=5)).date(),
                    max_value=default_end,
                )
            else:
                st.caption("Date window is fixed to latest 24h from local database.")
            selected_categories = st.multiselect(
                "Primary Categories",
                options=categories,
                default=categories,
            )
            bool_query = st.text_input("Boolean Search", value="")
            exact_filter_raw = st.text_input("Exact Filter Terms", value="")

            trend_terms_raw = default_terms
            granularity = "month"
            sort_key = "frontier"
            max_rows = 30000
            daily_show_code_only = False
            daily_cards_limit = 50

            if view == "Macro Trends":
                st.markdown("**Macro Trend Filters**")
                trend_terms_raw = st.text_input("Trend Terms", value=default_terms)
                granularity = st.selectbox("Trend Granularity", options=["month", "week", "day"], index=0)
            else:
                st.markdown("**Daily Brief Filters**")
                sort_key = st.selectbox(
                    "Ranking",
                    options=["newest", "open_source", "frontier"],
                    index=2,
                )
                max_rows = st.number_input(
                    "Max Rows",
                    min_value=5000,
                    max_value=300000,
                    value=30000,
                    step=5000,
                    help="Limit rows loaded into pandas for responsiveness.",
                )
                daily_show_code_only = st.checkbox("Show Code Only", value=False)
                daily_cards_limit = st.slider("Cards to display", min_value=10, max_value=200, value=50, step=10)
            st.form_submit_button("Apply Filters", use_container_width=True)
        if view == "Daily Brief":
            st.markdown("**Data Update**")
            st.caption("Independent of filter form. Pull latest papers then prune oldest by equal count.")
            update_pressed = st.button("Update", use_container_width=True)

    if view == "Macro Trends" and isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
    elif view == "Macro Trends":
        start_date = default_start
        end_date = default_end
    else:
        latest_raw = get_latest_published_at(conn)
        if latest_raw:
            try:
                latest_dt = datetime.fromisoformat(latest_raw)
            except ValueError:
                latest_dt = now
            else:
                if latest_dt.tzinfo is None:
                    latest_dt = latest_dt.replace(tzinfo=timezone.utc)
                latest_dt = latest_dt.astimezone(timezone.utc)
        else:
            latest_dt = now
        end_date = latest_dt
        start_date = latest_dt - timedelta(hours=24)

    exact_filter_terms = parse_terms_input(exact_filter_raw)
    trend_terms = parse_terms_input(trend_terms_raw)
    if not trend_terms:
        trend_terms = ["Diffusion", "LLM", "Agent"]
    effective_categories = selected_categories if selected_categories else categories

    if view == "Daily Brief":
        st.session_state["daily_show_code_only"] = bool(daily_show_code_only)
        st.session_state["daily_cards_limit"] = int(daily_cards_limit)
        if update_pressed:
            with st.spinner("Updating Daily Brief from arXiv..."):
                update_info = run_daily_from_latest_with_prune(
                    conn,
                    categories=categories,
                    prune_equal_inserted=True,
                )
                enriched_count = 0
                if int(update_info.get("inserted", 0)) > 0:
                    enriched_count = run_sync_for_arxiv_ids(
                        update_info.get("inserted_arxiv_ids", []),
                        progress_every=25,
                        recompute=False,
                        conn=conn,
                    )
            st.success(
                "Update completed: "
                f"processed={update_info['processed']}, "
                f"inserted={update_info['inserted']}, "
                f"deleted_oldest={update_info['deleted']}, "
                f"enriched_stars={enriched_count}"
            )
            st.caption(f"Window: {update_info['since']} -> {update_info['until']}")
            if update_info["failed_categories"]:
                failed = ", ".join(update_info["failed_categories"])
                st.warning(f"Some categories failed during update: {failed}")

    total_matches = count_papers(
        conn,
        start_date=start_date,
        end_date=end_date,
        categories=effective_categories,
        boolean_query=bool_query,
        exact_terms=exact_filter_terms,
        has_code_only=(view == "Daily Brief" and bool(daily_show_code_only)),
    )
    if view == "Daily Brief":
        st.caption(
            f"Daily window (UTC): {start_date.strftime('%Y-%m-%d %H:%M')} -> {end_date.strftime('%Y-%m-%d %H:%M')}"
        )
    st.caption(f"Matched papers: {total_matches:,}")

    if view == "Macro Trends":
        selected_days = (end_date - start_date).days + 1
        if granularity == "month" and selected_days <= 45:
            st.caption("Current date range is short; `month` granularity may collapse into one point. Try `week` or `day`.")
        elif granularity == "week" and selected_days <= 14:
            st.caption("Current date range is very short; `week` granularity may collapse into one point. Try `day`.")
        render_macro_trends(
            conn=conn,
            start_date=start_date,
            end_date=end_date,
            selected_categories=effective_categories,
            bool_query=bool_query,
            exact_filter_terms=exact_filter_terms,
            trend_terms=trend_terms,
            granularity=granularity,
        )
    elif view == "Daily Brief":
        if total_matches > int(max_rows):
            st.warning(
                f"Matched {total_matches:,} papers, capped to latest {int(max_rows):,} for UI performance. "
                "Increase `Max Rows` if needed."
            )
        with st.spinner("Loading papers from SQLite..."):
            df = query_papers(
                conn,
                start_date=start_date,
                end_date=end_date,
                categories=effective_categories,
                boolean_query=bool_query,
                exact_terms=exact_filter_terms,
                has_code_only=bool(daily_show_code_only),
                sort_key=sort_key,
                max_rows=int(max_rows),
            )
        st.caption(f"Loaded papers: {len(df):,} / matched: {total_matches:,}")
        render_daily_brief(df=df)


if __name__ == "__main__":
    main()
