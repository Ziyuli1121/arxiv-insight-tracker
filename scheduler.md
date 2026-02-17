# Daily Scheduler Template

Use cron to run one daily ingestion + enrichment cycle every 24 hours.

```cron
0 2 * * * cd /home/perry/ziyul6/arxiv_trend_tracker && /usr/bin/python ingest.py daily --window-hours 24 --categories cs.AI cs.LG cs.CV cs.CL && /usr/bin/python enrich.py sync --limit 300 >> logs/daily.log 2>&1
```

Adjust:

- Python path (`/usr/bin/python`)
- project path (`/home/perry/ziyul6/arxiv_trend_tracker`)
- schedule time (`0 2 * * *`)
