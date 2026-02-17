from __future__ import annotations

import pandas as pd

from processor import build_ai4s_df, build_combined_fts_query, extract_github_links, filter_by_boolean_query


def test_extract_github_links_deduplicates_and_normalizes():
    text = (
        "Code: https://github.com/org/repo and mirror "
        "http://github.com/org/repo plus https://github.com/user/another."
    )
    links = extract_github_links(text)
    assert links == ["https://github.com/org/repo", "https://github.com/user/another"]


def test_boolean_filter_supports_and_or_not():
    df = pd.DataFrame(
        [
            {"title": "Generative Biology", "abstract": "diffusion model for biology"},
            {"title": "Generative Vision", "abstract": "diffusion model for images"},
            {"title": "Symbolic Biology", "abstract": "logic methods for biology"},
        ]
    )
    df["_search_text"] = (df["title"] + " " + df["abstract"]).str.lower()

    filtered = filter_by_boolean_query(df, '"Generative" AND "Biology"')
    assert len(filtered) == 1
    assert filtered.iloc[0]["title"] == "Generative Biology"

    filtered = filter_by_boolean_query(df, '"Biology" AND NOT "diffusion"')
    assert len(filtered) == 1
    assert filtered.iloc[0]["title"] == "Symbolic Biology"


def test_build_ai4s_df_requires_cs_and_cross_domain():
    df = pd.DataFrame(
        [
            {"arxiv_id": "1", "categories": ["cs.AI", "physics.comp-ph"]},
            {"arxiv_id": "2", "categories": ["cs.LG"]},
            {"arxiv_id": "3", "categories": ["math.OC", "physics.optics"]},
            {"arxiv_id": "4", "categories": ["cs.CV", "q-bio.QM"]},
        ]
    )
    out = build_ai4s_df(df, cross_prefixes=["physics", "q-bio", "math"])
    assert sorted(out["arxiv_id"].tolist()) == ["1", "4"]


def test_build_combined_fts_query_enforces_exact_phrases():
    q = build_combined_fts_query(
        boolean_query='"Generative" AND Biology',
        exact_terms=["diffusion", "multi modal"],
    )
    assert q is not None
    assert '"Generative"' in q
    assert '"Biology"' in q
    assert '"diffusion"' in q
    assert '"multi modal"' in q
