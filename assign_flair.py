#!/usr/bin/env python3
"""
Assign Reddit flair (cluster) to rows missing it.

Expects sentiment pipeline output: rows with type (post/comment), cluster, text/clean_text.
Pipeline: run after sentiment.py. Input = sentiment_output.csv, output = labelled data.
TODO production: replace CSV paths with reads/writes to backend data tables.

--------------------------------------------------------------------------------
HOW EACH ROW GETS A CLUSTER (flair)
--------------------------------------------------------------------------------
- Rows that already have a valid cluster (non-empty, not "Unspecified") are left
  unchanged.

- POSTS missing cluster:
  We call Gemini once per post with the list of allowed flairs; it returns one
  flair (e.g. "Customer Support"). We write that into cluster.

- COMMENTS missing cluster:
  We do NOT call Gemini for comments. We look up the comment's parent post (via
  URL) in our data. If that post has a cluster, we assign the same cluster to the
  comment (inherited). If the post isn't in our set or has no cluster, we assign
  "Unmappable".

So: posts get LLM classification; comments get inherited flair or Unmappable.
"""
import os
import re
import json
import time
import pandas as pd
from pathlib import Path
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass
from google import genai

base = Path(__file__).parent
# Input: sentiment script output. TODO production: read from backend table.
input_csv = "sentiment_output.csv"
# Output: same rows with cluster filled. TODO production: write to backend table.
output_csv = "labelled_sentiment_output.csv"
UNMAPPABLE = "Unmappable"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set.")
print("Using Gemini key prefix:", GEMINI_API_KEY[:6], "…")
_flair_client = genai.Client(api_key=GEMINI_API_KEY)


FLAIRS = [
    "Stock Trading", "Option Trading", "Registered Accounts",
    "Cash & Margin Accounts", "Customer Support", "Feedback",
    "Taxes", "General", "Mobile apps", "Funding",
    "Transfers", "Web trading", "Edge desktop"
]

MAX_CLASSIFY_RETRIES = 5
POST_ID_RE = re.compile(r"/comments/([^/]+)/")


def classify_cluster(text: str) -> str:
    """Use Gemini to pick one FLAIRS value for post text; retry on parse failure; default to General."""
    prompt = (
        "You are a Reddit flair classifier. Available flairs are: "
        + ", ".join(FLAIRS)
        + ".\nGiven this post text, choose the most appropriate single flair."
        + "\nOutput _only_ a JSON object with one key \"cluster\" whose value is exactly one of the options.\n"
        + f"Text:\n\"\"\"{text}\"\"\""
    )
    for attempt in range(1, MAX_CLASSIFY_RETRIES + 1):
        try:
            resp = _flair_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            raw = (getattr(resp, "text", None) or "").strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                raise ValueError(f"No JSON in response: {raw!r}")
            obj = json.loads(m.group())
            cluster = obj.get("cluster")
            if cluster not in FLAIRS:
                raise ValueError(f"Invalid flair '{cluster}'")
            return cluster
        except Exception as e:
            print(f"⚠️ Flair classify error on attempt {attempt}: {e}")
            time.sleep(0.5)
    print(f"🚨 Defaulting to 'General' after {MAX_CLASSIFY_RETRIES} retries.")
    return "General"


def _has_cluster(val) -> bool:
    """True if cluster is non-null, non-blank, and not Unspecified."""
    if pd.isna(val):
        return False
    s = str(val).strip()
    return s != "" and s != "Unspecified"


def _post_id_from_url(url) -> str | None:
    """Extract Reddit post id from comment/post URL (e.g. .../comments/abc123/...)."""
    if pd.isna(url):
        return None
    m = POST_ID_RE.search(str(url))
    return m.group(1) if m else None


def main():
    # --- Load sentiment output (must have type and cluster columns) ---
    input_path = base / input_csv
    sample = pd.read_csv(input_path, nrows=0)
    parse_dates = ["date_utc"] if "date_utc" in sample.columns else None
    df = pd.read_csv(input_path, parse_dates=parse_dates)

    if "type" not in df.columns:
        raise ValueError("Input CSV must have a 'type' column (post vs comment).")

    # --- Partition: which rows already have cluster vs missing; which are post vs comment ---
    has_cluster = df["cluster"].map(_has_cluster)
    missing = ~has_cluster
    is_post, is_comment = df["type"] == "post", df["type"] == "comment"

    missing_posts = df.index[is_post & missing]
    n_posts = len(missing_posts)
    print(f"Posts with cluster: {(is_post & has_cluster).sum()}. Posts to label (LLM): {n_posts}.\n")

    # --- Fill missing clusters for POSTs via Gemini (one call per post) ---
    for i, idx in enumerate(missing_posts, 1):
        text = (df.at[idx, "clean_text"] if "clean_text" in df.columns else df.at[idx, "text"]) or ""
        assigned = classify_cluster(text)
        df.at[idx, "cluster"] = assigned
        row_id = df.at[idx, "id"] if "id" in df.columns else ""
        print(f"[post {i}/{n_posts}] ROW: id={row_id} | text={str(text)[:100]}{'…' if len(str(text)) > 100 else ''}")
        print(f"         ASSIGNED: {assigned} (via LLM)\n")
        time.sleep(0.1)

    # --- Build post id -> cluster so we can assign comments from their parent post ---
    post_id_to_cluster = {}
    for _, row in df[is_post].iterrows():
        pid = _post_id_from_url(row.get("url"))
        if pid and _has_cluster(row.get("cluster")):
            post_id_to_cluster[pid] = row["cluster"]

    missing_comments = df.index[is_comment & missing]
    n_comments = len(missing_comments)
    print(f"Comments with cluster: {(is_comment & has_cluster).sum()}. Comments to assign: {n_comments} (inherit or Unmappable).\n")

    # --- Fill missing clusters for COMMENTs: inherit from post or Unmappable (no LLM) ---
    for i, idx in enumerate(missing_comments, 1):
        text = (df.at[idx, "clean_text"] if "clean_text" in df.columns else df.at[idx, "text"]) or ""
        row_id = df.at[idx, "id"] if "id" in df.columns else ""
        pid = _post_id_from_url(df.at[idx, "url"])
        if pid and pid in post_id_to_cluster:
            assigned = post_id_to_cluster[pid]
            df.at[idx, "cluster"] = assigned
            print(f"[comment {i}/{n_comments}] ROW: id={row_id} | text={str(text)[:100]}{'…' if len(str(text)) > 100 else ''}")
            print(f"         ASSIGNED: {assigned} (inherited from post {pid})\n")
        else:
            df.at[idx, "cluster"] = UNMAPPABLE
            print(f"[comment {i}/{n_comments}] ROW: id={row_id} | text={str(text)[:100]}{'…' if len(str(text)) > 100 else ''}")
            print(f"         ASSIGNED: {UNMAPPABLE} (post not in data or no cluster)\n")
        time.sleep(0.05)

    # --- Write output (same rows, cluster column now filled; TODO production: backend table) ---
    out_path = base / output_csv
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path} ({len(df)} rows).")


if __name__ == "__main__":
    main()
