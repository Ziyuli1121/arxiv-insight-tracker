# ArXiv Insight Tracker

ArXiv Insight Tracker is a local-first research intelligence app built for fast, practical signal extraction from AI papers.

It is designed to answer two high-value questions every day:

- What directions are rising or fading over time?
- What are the newest papers worth reading right now?

## Why This Tracker Is Innovative

This project is not just a paper viewer. It combines historical context, daily monitoring, and practical ranking in one workflow:

- 5-year rolling AI knowledge base across core categories (`cs.AI`, `cs.CL`, `cs.CV`, `cs.LG`)
- Exact keyword and boolean retrieval for reliable trend analysis
- Frontier-oriented ranking (not only recency) to surface high-signal papers
- Built-in open-source awareness through GitHub link extraction and stars enrichment
- One-click daily refresh in UI with rolling-database maintenance (`insert X`, prune oldest `X`)

## Core Product Capabilities

### 1) Macro Trends

- Time-window analysis (default: last 12 months)
- Multi-term trend lines (exact match semantics)
- Category share streamgraph
- Boolean search (`AND` / `OR` / `NOT`, phrase support)
- Optional keyword momentum computation (slower, deeper insight)

### 2) Daily Brief

- Fixed to the latest 24h window based on your local database head
- Fast ranking by:
  - `newest`
  - `open_source`
  - `frontier`
- `Show Code Only` filter
- Paper cards with:
  - title / authors / publish time / category
  - open-source metrics
  - collapsible abstract for quick reading
- `Update` button:
  - fetches increment from latest local `published_at`
  - prunes oldest papers by equal inserted count
  - enriches newly inserted papers with GitHub stars

## Stack

- Python 3.9+
- Streamlit (UI)
- SQLite (storage)
- Pandas + Plotly (analytics/visualization)
- `arxiv` (data source)
- scikit-learn (TF-IDF keyword processing)

## Quick Start

### 1) Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Build the 5-year base corpus

```bash
python ingest.py init --years 5 --categories cs.AI cs.LG cs.CV cs.CL
```

Notes:

- Backfill is chunked by `category x month`.
- It is resumable via checkpoint state in `pipeline_state`.
- First run can be long due to arXiv rate limits.

### 3) Enrich open-source metadata

Single batch:

```bash
python enrich.py sync --limit 300
```

Recommended for initial large pass:

```bash
python enrich.py sync --limit 1000 --progress-every 20 --no-recompute
```

Then recompute scores once:

```bash
python enrich.py recompute
```

### 4) Run the app

```bash
streamlit run app.py
```

## Daily Operations

### CLI mode (scheduled)

```bash
python ingest.py daily --window-hours 24 --categories cs.AI cs.LG cs.CV cs.CL
python enrich.py sync --limit 300
```

### UI mode (interactive)

- Open `Daily Brief`
- Click `Update`

## Running Long Jobs in tmux

Backfill:

```bash
tmux new -s arxiv
python ingest.py init --years 5 --categories cs.AI cs.LG cs.CV cs.CL 2>&1 | tee logs/init_$(date +%F_%H-%M).log
```

Bulk enrich:

```bash
python enrich.py sync --limit 1000 --progress-every 20 --no-recompute 2>&1 | tee -a logs/enrich_$(date +%F).log
```

Final score recompute:

```bash
python enrich.py recompute
```

## Recommended 24h Cron

```cron
0 2 * * * cd /home/perry/ziyul6/arxiv_trend_tracker && /usr/bin/python ingest.py daily --window-hours 24 --categories cs.AI cs.LG cs.CV cs.CL && /usr/bin/python enrich.py sync --limit 300 >> logs/daily.log 2>&1
```

## Key Config Variables

- `ARXIV_TRACKER_DB_PATH`: SQLite path (default `data/arxiv_insight.db`)
- `ARXIV_DELAY_SECONDS`: delay between arXiv API calls
- `ARXIV_NUM_RETRIES`: arXiv retry count
- `ARXIV_PAGE_SIZE`: API page size
- `GITHUB_TOKEN`: GitHub token for stars enrichment
- `GITHUB_DELAY_SECONDS`: delay between GitHub API calls

## Scoring (Current)

- `open_source_score`: based on `has_code` and normalized GitHub stars
- `frontier_score`: weighted sum of:
  - `keyword_burst`
  - `open_source_score`
  - `freshness`

These scores are intended for prioritization, not absolute scientific quality judgment.

## Troubleshooting

### `sqlite3.OperationalError: database is locked`

- Ensure you do not run multiple writers on the same DB at the same time.
- Stop stale ingestion/enrichment processes and restart one pipeline.

### `No papers pending enrichment`

- Means there are no currently eligible `has_code=1` rows awaiting enrichment.

### App feels slow

- Narrow date window for Macro Trends.
- Reduce `Max Rows` in Daily Brief.
- Complete base ingest/enrich first, then do broad analysis.

## Project Layout

- `app.py`: Streamlit UI
- `ingest.py`: arXiv ingestion (5-year backfill + daily increment)
- `enrich.py`: GitHub stars enrichment + recompute
- `processor.py`: querying, aggregation, trend/scoring logic
- `db.py`: schema, indexes, upserts
- `config.py`: runtime configuration
- `tests/`: unit tests
