# Questrade Reddit Pipeline

This document describes the three scripts that make up the Reddit pipeline: **scraper** → **sentiment** → **assign flair**. Each script reads from the previous step’s output (or from Reddit) and writes the input for the next step. The goal is to collect r/Questrade posts and comments, score sentiment toward Questrade, and assign a topic flair (cluster) to each row.

---

## 1. Overview and pipeline order

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `Updated Scrapers/reddit_network_old.py` | Fetches posts and comments from Reddit in a date range; writes one CSV with `type` (post/comment), `cluster` from Reddit flair (or "Unspecified"). |
| 2 | `Data Cleaning & Sentiment/sentiment.py` | Loads that CSV, cleans text, tags competitor mentions, and assigns a **sentiment** score per row using FinBERT and/or Gemini. Writes a new CSV with a `sentiment` column. |
| 3 | `Data Cleaning & Sentiment/assign_flair.py` | Loads the sentiment CSV, fills in missing **cluster** (flair): posts via Gemini; comments by inheriting from their post or "Unmappable". Writes the final labelled CSV. |

You must run them in this order. Step 2 expects the scraper’s CSV; step 3 expects the sentiment script’s CSV.

---

## 2. Script 1: Reddit scraper (`reddit_network_old.py`)

**Location:** `Questrade/Updated Scrapers/reddit_network_old.py`

### What it does

1. **Resolves a date range** (UTC):
   - If you pass both `--start` and `--end` (ISO dates, e.g. `2025-02-25`), that window is used.
   - If you omit them, the script uses “previous day(s)” in local time (`SCRAPER_TZ`, default `America/Toronto`):  
     - Usually **yesterday** (one day).  
     - On **Monday** it uses Friday 00:00 → Monday 00:00 (three days) so the weekend is included if the job didn’t run Sat/Sun.

2. **Fetches posts** from `https://old.reddit.com/r/{sub}/new/.json`:
   - Newest first, 100 per page, paginating with Reddit’s `after` cursor.
   - Keeps only posts whose `created_utc` is in `[start_ts, end_ts)`.
   - Stops when it sees a post older than `start_ts` (because listing is newest-first).
   - For each post: `date_utc` (converted to local time), `text` (title + selftext), `cluster` (link flair or "Unspecified"), `upvotes`, `author`, `id`, `url`.

3. **Fetches comments** for each post from `https://old.reddit.com/r/{sub}/comments/{id}.json`:
   - Uses the first comment listing (index `[1]` in the JSON; `[0]` is the post).
   - Keeps only items with `kind == "t1"` (actual comments; skips “more” placeholders).
   - Each comment gets the same fields; `cluster` is copied from the parent post.

4. **Combines and writes:**
   - Adds a column `type`: `"post"` or `"comment"`.
   - Builds one DataFrame from all posts and comments, sorts by `date_utc`, writes CSV (default: `questrade_recent.csv` in the same folder as the script).

### Structure (main pieces)

- **Constants:** `USER_AGENT`, `LOCAL_TZ`, `BASE_POST`, `BASE_COMM`, `MAX_RETRIES`, `POST_SLEEP`, `COMMENT_SLEEP`, `BACKOFF_BASE`.
- **`ts_from_str(s)`** — Parses an ISO date string to a UTC timestamp (seconds).
- **`_ts_to_local(ts)`** — Converts a UTC timestamp to local time (for `date_utc` in the CSV).
- **`get_with_backoff(session, url, params)`** — GET with retries; on 429, waits using `Retry-After` or exponential backoff.
- **`fetch_posts(sub, start_ts, end_ts, pulse=10)`** — Paginates `/new/.json`, collects posts in range, returns list of dicts.
- **`fetch_comments(sub, posts, pulse=10)`** — For each post, fetches comment listing, returns list of comment dicts (with post’s `cluster`).
- **`get_previous_day_range_utc()`** — Returns `(start_dt, end_dt)` in UTC for “previous day(s)” (yesterday or Fri–Mon when today is Monday).
- **`if __name__ == "__main__"`** — Parses CLI, resolves date range, calls `fetch_posts` then `fetch_comments`, adds `type`, builds DataFrame, writes CSV.

### Output CSV columns

| Column     | Description |
|-----------|-------------|
| `date_utc`| Datetime in local time (SCRAPER_TZ). |
| `text`    | For posts: title + selftext; for comments: body (newlines replaced by space). |
| `cluster` | Reddit link flair for posts, or same as parent for comments; or "Unspecified". |
| `upvotes` | Score. |
| `author`  | Reddit username. |
| `url`     | Full Reddit URL (used later to get post id for comment inheritance). |
| `type`    | `"post"` or `"comment"`. |
| `id`      | Reddit id. |

### How to run

From the repo (or anywhere, if you set paths):

```bash
cd "Questrade/Updated Scrapers"
python reddit_network_old.py
```

Optional arguments:

- `--sub Questrade` — Subreddit (default: Questrade).
- `--start 2025-02-24 --end 2025-02-26` — Explicit date range (ISO). If you omit both, auto “previous day(s)” is used.
- `--out path/to/output.csv` — Output path (default: `questrade_recent.csv` in the script directory).

Example with explicit range:

```bash
python reddit_network_old.py --sub Questrade --start 2025-02-24 --end 2025-02-26 --out questrade_recent.csv
```

---

## 3. Script 2: Sentiment scoring (`sentiment.py`)

**Location:** `Questrade/Data Cleaning & Sentiment/sentiment.py`

### What it does

1. **Loads** the scraper CSV (path from `DATA_PATH`, default `questrade_recent.csv` in the same folder).
2. **Cleans and filters:**
   - Drops rows with `text` in `["[deleted]", "[removed]"]`.
   - Drops authors containing `"questrade"` and author `"automoderator"`.
   - Drops rows with empty or whitespace-only `text`.
   - Builds **`clean_text`**: lowercase, strip URLs, keep only `a-z0-9` and `.,!?$%`, strip, cast to string.
3. **Tags competitor mentions:** Adds **`competitors_mentioned`** (True/False) using regex for keywords (e.g. wealthsimple, ws, ibkr, robinhood, td, webull).
4. **Scores sentiment** for each row with a **FinBERT vs Gemini** decision tree (see below).
5. **Writes** the result to `sentiment_output.csv` (or path in `OUTPUT_CSV`) in the same folder.

### How each row gets a sentiment score (FinBERT vs Gemini)

For **every row** the script applies this logic:

1. **Does the text mention a competitor** (e.g. IBKR, Wealthsimple)?
   - **Yes** → Use **Gemini** with the **competitor** prompt (sentiment toward Questrade only). FinBERT is not used for these rows.

2. **No competitor mention** → Run **FinBERT** on `clean_text` to get positive / neutral / negative scores.
   - If FinBERT is **very confident the text is neutral** (neutral score ≥ `FINBERT_THRESHOLD`, default 0.8):  
     use **FinBERT’s** score = `positive - negative`. No Gemini call.
   - Otherwise (FinBERT says positive/negative or is unsure):  
     use **Gemini** with the **general** sentiment prompt. FinBERT is not Questrade-specific, so it’s only trusted when it strongly says “neutral”; everything else goes to the LLM.

**Summary:** Competitor mention → always Gemini (competitor prompt). No competitor → try FinBERT; accept FinBERT only when neutral ≥ threshold, else Gemini (general prompt). If FinBERT fails to load (e.g. missing `transformers`/`torch`), every row uses Gemini.

### Structure (main pieces)

- **Config:** `GEMINI_API_KEY`, `FINBERT_THRESHOLD`, `GEMINI_MODEL`, `DATA_PATH`, `OUTPUT_CSV`, `MAX_LLM_RETRIES`. Optional load of `.env` via `python-dotenv`.
- **Competitor list:** `COMPETITORS` dict and `COMPETITOR_PATTERNS` (regexes). Used to set `competitors_mentioned`.
- **FinBERT:** Optional `transformers` pipeline (`ProsusAI/finbert`, `return_all_scores=True`). On import failure, `finbert` is `None` and every row uses Gemini.
- **`load_and_clean(path)`** — Load CSV, apply filters above, add `clean_text`.
- **`tag_competitors(df)`** — Add `competitors_mentioned`.
- **FinBERT helpers:** `_finbert_to_scores`, `_parse_finbert_output`, `_finbert_scores_via_model` (when pipeline returns only one label, get all three from model logits). **`finbert_score(text)`** — Returns sentiment (pos − neg) only when neutral ≥ threshold; else `None` (caller uses Gemini).
- **LLM:** `PROMPT_SENTIMENT`, `PROMPT_COMPETITOR`; **`_llm_sentiment_one(prompt, text, label)`** — Calls Gemini, parses JSON `"sentiment"`, retries on failure; **`llm_sentiment(text)`**, **`llm_competitor_sentiment(text)`**.
- **`hybrid_sentiment(row)`** — Implements the decision tree: competitor → LLM competitor; else FinBERT if not None, else LLM general.
- **`main()`** — Load and clean, tag competitors, `df["sentiment"] = df.apply(hybrid_sentiment, axis=1)`, write CSV.

### Input (from scraper CSV)

Expects at least: `date_utc`, `text`, `cluster`, `upvotes`, `author`, `url`, `type`, `id`. Path set by `DATA_PATH`.

### Output CSV

Same columns as input (after filtering), plus:

- **`clean_text`** — Normalized text used for FinBERT and Gemini.
- **`competitors_mentioned`** — Boolean.
- **`sentiment`** — Float in [-1, 1] (or 0.0 on LLM failure).

Default output file: `sentiment_output.csv` in `Data Cleaning & Sentiment/`.

### How to run

From the folder that contains the script and (if using default path) the scraper output:

```bash
cd "Questrade/Data Cleaning & Sentiment"
python sentiment.py
```

Ensure `GEMINI_API_KEY` is set (e.g. in `.env`). If the scraper wrote to a different path, set `DATA_PATH` (env or edit the constant) to that path (relative to the script folder or absolute).

---

## 4. Script 3: Assign flair (`assign_flair.py`)

**Location:** `Questrade/Data Cleaning & Sentiment/assign_flair.py`

### What it does

1. **Loads** the sentiment CSV (default: `sentiment_output.csv` in the same folder).
2. **Partitions rows:**
   - Rows that **already have a valid cluster** (non-null, non-blank, not `"Unspecified"`) are left unchanged.
   - **Posts** with missing cluster → classified with **Gemini** (one call per post); result written to `cluster`.
   - **Comments** with missing cluster → **no Gemini**; cluster is set by **inheriting** from the parent post (looked up by post id from `url`), or `"Unmappable"` if the post isn’t in the data or has no cluster.
3. **Writes** the same rows with `cluster` filled to `labelled_sentiment_output.csv` (or path in `output_csv`).

### How each row gets a cluster (flair)

- **Already has cluster** (valid value) → keep it.
- **Post, missing cluster** → Gemini is called with the list of allowed flairs; it returns one flair (e.g. `"Customer Support"`). That value is written to `cluster`. On repeated failure, default is `"General"`.
- **Comment, missing cluster** → Parent post id is parsed from the comment’s `url` (e.g. `/comments/abc123/...`). If that post is in the dataset and has a valid cluster, the comment gets that cluster (inherited). Otherwise the comment gets **`"Unmappable"`**.

So: **posts** get LLM classification; **comments** get inherited flair or Unmappable.

### Structure (main pieces)

- **Config:** `base` (script dir), `input_csv`, `output_csv`, `UNMAPPABLE`, `GEMINI_API_KEY`, `GEMINI_MODEL`. Optional `.env` load.
- **`FLAIRS`** — List of allowed flair strings (e.g. "Stock Trading", "Customer Support", "General", …). Gemini must return exactly one of these.
- **`classify_cluster(text)`** — Builds a prompt with `FLAIRS`, calls Gemini, parses JSON `"cluster"`, validates against `FLAIRS`, retries on failure; returns `"General"` after max retries.
- **`_has_cluster(val)`** — True if cluster is non-null, non-blank, and not `"Unspecified"`.
- **`_post_id_from_url(url)`** — Extracts post id from Reddit URL (e.g. `.../comments/abc123/...` → `abc123`).
- **`main()`** — Load CSV (with optional `date_utc` parsing), require `type` column; partition by `has_cluster` and `type`; fill missing posts via `classify_cluster`; build `post_id_to_cluster` from all posts; fill missing comments by inheritance or Unmappable; write CSV.

### Input (from sentiment CSV)

Must have: `type` (post/comment), `cluster`, and either `clean_text` or `text`. Uses `url` to get post id for comments, and `id` for logging. Optional: `date_utc` (parsed if present).

### Output CSV

Same rows and columns as input, with **`cluster`** filled for every row. Default file: `labelled_sentiment_output.csv` in `Data Cleaning & Sentiment/`.

### How to run

```bash
cd "Questrade/Data Cleaning & Sentiment"
python assign_flair.py
```

Requires `GEMINI_API_KEY`. Input path is set by `input_csv` (default `sentiment_output.csv` in the same folder); output by `output_csv` (default `labelled_sentiment_output.csv`).

---
