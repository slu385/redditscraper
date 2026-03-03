"""
Reddit scraper for r/Questrade: fetches posts and comments in a date range.

Pipeline: this script runs first; output is the input for sentiment.py.
In production, --out (and possibly date range) may be replaced by writing to
backend data tables / config instead of CLI and CSV.

--------------------------------------------------------------------------------
WHAT THIS SCRIPT DOES (high-level flow)
--------------------------------------------------------------------------------
1. Resolve date range
   - If you pass both --start and --end (ISO dates), we use that window.
   - If you omit them, we use "previous day(s)" in local time (SCRAPER_TZ):
     normally yesterday; on Monday we use Fri 00:00 -> Mon 00:00 so weekend is
     included if the job didn't run Sat/Sun.

2. Fetch posts
   - We call Reddit's /r/{sub}/new/.json (newest first, 100 per page).
   - We paginate with the "after" cursor until we see a post older than
     start_ts, and we only keep posts with created_utc in [start_ts, end_ts).
   - Each post gets: date_utc (in local time), text (title + selftext), flair
     as cluster, upvotes, author, id, url.

3. Fetch comments
   - For each post we call /r/{sub}/comments/{id}.json and take the top-level
     comment list. Comments get the same fields; cluster is copied from the post
     (Reddit flair is per-post; we inherit for comments).

4. Combine and write
   - We add a "type" column (post vs comment), merge posts + comments, sort
     by date_utc, and write one CSV (or in production, a table).
"""
import os
import time
import random
import argparse
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

SCRIPT_DIR = Path(__file__).resolve().parent

# Reddit API and rate-limiting
USER_AGENT = "windows:questrade_scraper:v1.3 (by /u/YourUsername)"
LOCAL_TZ = os.getenv("SCRAPER_TZ", "America/Toronto")
BASE_POST = "https://old.reddit.com/r/{sub}/new/.json"
BASE_COMM = "https://old.reddit.com/r/{sub}/comments/{id}.json"
MAX_RETRIES, POST_SLEEP = 10, (3.0, 7.0)
COMMENT_SLEEP, BACKOFF_BASE = (8.0, 15.0), 30


def ts_from_str(s):
    """Parse ISO date string to UTC timestamp (seconds)."""
    return int(datetime.fromisoformat(s)
               .replace(tzinfo=timezone.utc)
               .timestamp())


def _ts_to_local(ts: int):
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(ZoneInfo(LOCAL_TZ))


def get_with_backoff(session, url, params):
    """GET with retries; on 429, honor Retry-After or exponential backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        r = session.get(url, params=params)

        if r.status_code == 200:
            return r
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            if ra and ra.isdigit():
                wait = int(ra) + random.uniform(1, 5)
            else:
                wait = min(3600, BACKOFF_BASE * 2 ** (attempt - 1)) + random.uniform(1, 5)
            print(f"  429 on {url}; sleeping {wait:.0f}s (attempt {attempt})…")
            time.sleep(wait)
            continue

        print(f"  HTTP {r.status_code} on {url}; skipping this request")
        return None
    print(f"  Giving up after {MAX_RETRIES} retries on {url}")
    return None


def fetch_posts(sub, start_ts, end_ts, pulse=10):
    """Fetch posts from sub in [start_ts, end_ts) (UTC). Paginates via /new/.json."""
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    after, all_posts = None, []
    count = 0

    while True:
        params = {"limit": 100}
        if after:
            params["after"] = after  # Reddit pagination: next page after this fullname

        r = get_with_backoff(sess, BASE_POST.format(sub=sub), params)

        if not r:
            break

        data = r.json().get("data", {})
        batch = data.get("children", [])
        if not batch:
            break

        for item in batch:
            d, ts = item["data"], item["data"]["created_utc"]
            # /new/ is newest-first; once we're before start_ts we're done
            if ts < start_ts:
                return all_posts
            if ts >= end_ts:
                continue  # skip posts at or after end (e.g. today)

            all_posts.append({
                "date_utc": _ts_to_local(ts),
                "text":      (d.get("title","") + " " + d.get("selftext","")).strip(),
                "cluster":   d.get("link_flair_text") or "Unspecified",
                "upvotes":   d.get("score", 0),
                "author":    d.get("author", "[deleted]"),
                "id":        d["id"],
                "url":       "https://reddit.com" + d.get("permalink", "")
            })

            count += 1
            if count % pulse == 0:
                print(f"⏳ Fetched {count} posts so far, last id={d['id']}")

        after = data.get("after")  # None when no more pages
        if not after:
            break

        time.sleep(random.uniform(*POST_SLEEP))

    return all_posts


def fetch_comments(sub, posts, pulse=10):
    """Fetch top-level comments for each post. Comments inherit post flair from API."""
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    all_cm = []
    total_posts = len(posts)

    for idx, post in enumerate(posts, start=1):
        print(f"▶ [{idx}/{total_posts}] Fetching comments for post {post['id']}")
        url = BASE_COMM.format(sub=sub, id=post["id"])
        r   = get_with_backoff(sess, url, {"limit": 500})
        if not r:
            continue

        try:
            # Reddit comments response: [0] = post, [1] = comment listing
            cmts = r.json()[1]["data"]["children"]
        except Exception:
            print(f"  malformed comments for {post['id']}")
            continue

        count = 0
        for c in cmts:
            # skip “more” placeholders and any non-comment kinds
            if c.get("kind") != "t1":  # t1 = comment; skip "more" placeholders
                continue
            d = c["data"]
            if "created_utc" not in d:
                continue

            all_cm.append({
                "date_utc": _ts_to_local(d["created_utc"]),
                "text":      d.get("body","").replace("\n"," "),
                "cluster":   post["cluster"],
                "upvotes":   d.get("score", 0),
                "author":    d.get("author","[deleted]"),
                "id":        d["id"],
                "url":       f"https://reddit.com/r/{sub}/comments/{post['id']}/_/{d['id']}"
            })

            count += 1
            if count % pulse == 0:
                print(f"    ⏳ Fetched {count} comments for post {post['id']}")

        time.sleep(random.uniform(*COMMENT_SLEEP))

    return all_cm


def get_previous_day_range_utc():
    """Return (start, end) in UTC for 'previous day(s)': yesterday, or Fri–Mon if today is Monday."""
    tz = ZoneInfo(LOCAL_TZ)
    now = datetime.now(tz)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # Monday = 0: include Fri/Sat/Sun so weekend isn't missed if job didn't run
    days_back = 3 if now.weekday() == 0 else 1
    start = (today - timedelta(days=days_back)).astimezone(timezone.utc)
    end = today.astimezone(timezone.utc)
    return start, end


if __name__ == "__main__":
    # CLI is used for ad-hoc/scheduled runs. In production, consider env/config for sub, date range, and output target (e.g. table sink).
    p = argparse.ArgumentParser(
        description="Scrape Reddit posts/comments. Date range: use --start and --end together for explicit range, or omit both for auto (previous day(s); Monday = Fri+Sat+Sun)."
    )
    p.add_argument("--sub",   default="Questrade", help="Subreddit name")
    p.add_argument("--start", default=None, help="Start date (ISO), e.g. 2025-02-25. Must set both --start and --end for explicit range.")
    p.add_argument("--end",   default=None, help="End date (ISO). Must set both --start and --end for explicit range.")
    p.add_argument("--out",   default=str(SCRIPT_DIR / "questrade_recent.csv"), help="Output path. TODO: replace with backend table write in production.")
    args = p.parse_args()

    # --- Step 1: Resolve date range (explicit or previous day(s)) ---
    if args.start and args.end:
        start_ts = ts_from_str(args.start)
        end_ts = ts_from_str(args.end)
        print(f"Using date range: {args.start} -> {args.end}")
    else:
        start_dt, end_dt = get_previous_day_range_utc()
        start_ts = int(start_dt.timestamp())
        end_ts   = int(end_dt.timestamp())
        print(f"Auto date range (previous day(s), {LOCAL_TZ}): {start_dt.isoformat()} -> {end_dt.isoformat()} UTC")

    # --- Step 2: Fetch posts in range (paginate /new/.json until we pass start_ts) ---
    print("▶ Fetching posts…")
    posts = fetch_posts(args.sub, start_ts, end_ts)
    print(f"  Retrieved {len(posts)} posts")

    # --- Step 3: For each post, fetch its top-level comments ---
    print("▶ Fetching comments…")
    comments = fetch_comments(args.sub, posts)
    print(f"  Retrieved {len(comments)} comments")

    # --- Step 4: Tag type (post vs comment) and combine into one table ---
    for p in posts:
        p["type"] = "post"
    for c in comments:
        c["type"] = "comment"

    df = pd.DataFrame(
        posts + comments,
        columns=["date_utc","text","cluster","upvotes","author","url","type","id"]
    )
    df.sort_values("date_utc", inplace=True)
    # --- Step 5: Write output (TODO production: backend table) ---
    out_path = Path(args.out) if os.path.isabs(args.out) else SCRIPT_DIR / args.out
    df.to_csv(out_path, index=False)
    print("✔ Saved", len(df), "rows to", out_path)
