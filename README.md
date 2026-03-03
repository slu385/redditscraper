# Questrade Reddit Pipeline

## Pipeline Overview

Three scripts run in sequence: **Scraper → Sentiment → Assign Flair**.

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 — Scrape | `Updated Scrapers/reddit_network_old.py` | Reddit API (r/Questrade) | `questrade_recent.csv` |
| 2 — Sentiment | `Data Cleaning & Sentiment/sentiment.py` | Scraper CSV | `sentiment_output.csv` |
| 3 — Flair | `Data Cleaning & Sentiment/assign_flair.py` | Sentiment CSV | `labelled_sentiment_output.csv` |

> Steps 2 and 3 are interchangeable in the target state once tables are implemented.

---

## Step 1: Scraper (`reddit_network_old.py`)

Fetches posts and comments from `r/Questrade` within a date range (UTC).

**Date resolution:** Pass `--start` / `--end` (ISO dates), or omit for auto — yesterday by default, Fri–Mon on Mondays (covers the weekend). In production I imagine this just pulling posts from the previous day.

**Process:** Paginates `/new/.json` (newest-first, 100/page) → keeps posts in `[start, end)` → fetches each post's comment listing → combines into one CSV with a `type` column (`post` | `comment`).

**Key functions:** `get_with_backoff` (retries + 429 handling), `fetch_posts`, `fetch_comments`, `get_previous_day_range_utc`.

**Output columns:** `date_utc` (local TZ), `text`, `cluster` (Reddit flair or "Unspecified" - comments receive same flair as parent post), `upvotes`, `author`, `id`, `url`, `type`.

---

## Step 2: Sentiment (`sentiment.py`)

Cleans text, tags competitor mentions, and assigns a sentiment score per row. FinBERT model is used to assign positive, neutral, negative score. If neutral score is below a specific threshold, OR a competitor is mentioned, the text is passed to an LLM for a better sentiment score.

**Cleaning:** Drops deleted/removed/bot rows → builds `clean_text` (lowercase, strip URLs, alphanumeric + punctuation only).

**Competitor tagging:** Regex match for keywords (wealthsimple, ibkr, robinhood, td, webull, etc.) → `competitors_mentioned` (bool).

**Sentiment decision tree:**

| Condition | Method | Prompt |
|-----------|--------|--------|
| Competitor mentioned | Gemini | Competitor prompt (QT-only sentiment) |
| No competitor, FinBERT neutral ≥ 0.8 | FinBERT | `pos - neg` score |
| No competitor, FinBERT unsure/polar | Gemini | General prompt |
| FinBERT unavailable | Gemini | Always |

**Added columns:** `clean_text`, `competitors_mentioned`, `sentiment` (float in [-1, 1], 0.0 on failure).

---

## Step 3: Assign Flair (`assign_flair.py`)

Fills missing `cluster` values using Gemini (posts) or inheritance (comments).

| Row state | Action |
|-----------|--------|
| Already has valid cluster | Keep as-is |
| Post, missing cluster | Gemini classifies from allowed `FLAIRS` list (default fallback: "General") |
| Comment, missing cluster | Inherits parent post's cluster via URL parsing, or "Unmappable" |

**Output:** Same columns with `cluster` filled for every row → `labelled_sentiment_output.csv`.

---

## Next Steps

### 1. Migrate from CSVs to SQL Server tables

Replace CSV hand-offs with persistent tables accessible through MSSQL Server Management Studio. Two separate table sets needed — one for **Questrade (QT)** and one for **Wealthsimple (WS)**.

**Per subreddit (QT / WS):**

| Table | Purpose | Notes |
|-------|---------|-------|
| `scraped_data` | Raw posts and comments (historical) | Append-only; optional snapshot/temp table between steps if helpful |
| `sentiment_labelled` | Sentiment scores + cluster assignments (historical) | Needs a defined archive/retention cadence |

> Since steps 2 and 3 are interchangeable, the pipeline could run flair assignment before sentiment if that simplifies the table design. If necessary, we can use a third snapshot table in-between the scraped and sentiment tables for flair assignment.

### 2. API key → Application Default Credentials

Move away from `.env`-stored API keys (Gemini, etc.) to Application Default Credentials (ADC) for all service authentication.

### 3. Airflow DAGs (QT + WS)

Two DAGs — one per subreddit — scheduled to run every morning:

1. **Scrape task** — fetch posts and comments, write to `scraped_data` table
2. **Sentiment task** — score rows, write to `sentiment_labelled` table
3. **Cluster task** — assign flair, update `sentiment_labelled` table

### 4. LangSmith for evals and observability

Integrate LangSmith to track LLM calls (Gemini sentiment + flair classification), run evaluations, and monitor prompt performance over time.

### 5. Housekeeping

- **`requirements.txt`** — pin all dependencies and integrate into the build/deploy process
- **`.env` cleanup** — audit and remove any secrets that should live in ADC or a secret manager; keep `.env` for non-sensitive config only
- **PaaS config practices** — adopt platform-as-a-service configuration standards (environment-based config, secret management, health checks, logging)
