#!/usr/bin/env python3
"""
Sentiment scoring for Questrade Reddit pipeline.

Reads scraper output (posts + comments), cleans text, tags competitor mentions,
then scores sentiment: FinBERT when confident neutral, else Gemini LLM.
Pipeline: run after reddit_network_old.py; output is input for assign_flair.py.
TODO production: replace CSV paths with reads/writes to backend data tables.

--------------------------------------------------------------------------------
HOW EACH ROW GETS A SENTIMENT SCORE (FinBERT vs Gemini)
--------------------------------------------------------------------------------
For every row we run this decision tree:

  1. Does the text mention a competitor (e.g. IBKR, Wealthsimple)?
     YES → Use GEMINI with the "competitor" prompt (rates sentiment toward
           Questrade only). We never use FinBERT for competitor mentions.

  2. No competitor mention → Run FinBERT to get positive/neutral/negative scores.
     - If FinBERT is very confident the text is NEUTRAL (neutral score >= 0.8):
       use FinBERT's (positive - negative) as the sentiment. Cheap, no API call.
     - Otherwise (FinBERT thinks it's positive or negative, or is unsure):
       use GEMINI with the general sentiment prompt. FinBERT is not Questrade-
       specific and can misread tone, so we only trust it when it strongly says
       "neutral"; everything else goes to the LLM.

So: competitor → always Gemini. No competitor → try FinBERT; only accept FinBERT
when neutral >= threshold, else Gemini.
"""
import os
import re
import json
import pandas as pd
from pathlib import Path
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass
from google import genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
FINBERT_THRESHOLD = float(os.getenv("FINBERT_THRESHOLD", 0.8))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
# Input: scraper output. TODO production: read from backend table.
# Input: scraper output (posts + comments). TODO production: read from backend table.
DATA_PATH = "questrade_recent.csv"
# Output: same rows + sentiment column for assign_flair. TODO production: write to backend table.
OUTPUT_CSV = "sentiment_output.csv"
MAX_LLM_RETRIES = 5

# FinBERT is optional (transformers/torch); if it fails to load, every row uses Gemini.
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set.")
print("Using Gemini key prefix:", GEMINI_API_KEY[:6], "…")
_llm_client = genai.Client(api_key=GEMINI_API_KEY)

# Competitor keywords: if present, we use a dedicated LLM prompt for Questrade-vs-competitor sentiment.
COMPETITORS = {
    "wealthsimple": ["wealthsimple", "ws"],
    "ibkr":         ["ibkr", "interactive brokers"],
    "robinhood":    ["robinhood"],
    "td":           ["td", "td ameritrade"],
    "webull":       ["webull"]
}

COMPETITOR_PATTERNS = {
    name: re.compile(r"\b(?:" + "|".join(map(re.escape, kws)) + r")\b")
    for name, kws in COMPETITORS.items()
}

try:
    from transformers import pipeline
    import torch
    finbert = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        return_all_scores=True
    )
    print("FinBERT loaded (all scores).")
except Exception:
    finbert = None
    torch = None
    print("⚠️  FinBERT unavailable — falling back to LLM for every row.")


# ----------------------------------------------------------------------------
# Data loading and competitor tagging (run once before scoring)
# ----------------------------------------------------------------------------

def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV from path (relative to this script or absolute), drop deleted/removed/Questrade authors, normalize text to clean_text."""
    base = Path(__file__).parent
    fp   = Path(path) if Path(path).is_absolute() else base / path
    if not fp.exists():
        raise FileNotFoundError(f"Missing input file: {fp}")

    df = pd.read_csv(fp, parse_dates=["date_utc"])
    # Drop deleted/removed content and Questrade/official authors so we don't score their own posts
    df = df[~df.text.isin(["[deleted]", "[removed]"])]
    df["author"] = df["author"].str.lower().fillna("")
    df = df[~df["author"].str.contains("questrade")]
    df = df[~df.author.eq("automoderator")]
    df = df[df.text.str.strip().astype(bool)].reset_index(drop=True)
    # Normalize for FinBERT/LLM: lowercase, strip URLs and non-alpha, collapse spaces
    df["clean_text"] = (
        df.text
          .str.lower()
          .str.replace(r"http\S+", "", regex=True)
          .str.replace(r"[^a-z0-9\s\.,!?$%]", "", regex=True)
          .str.strip()
          .fillna("")
          .astype(str)
    )
    return df


def tag_competitors(df: pd.DataFrame) -> pd.DataFrame:
    """Set competitors_mentioned from COMPETITOR_PATTERNS so we can branch to competitor LLM prompt."""
    df["competitors_mentioned"] = df.clean_text.apply(
        lambda txt: any(pat.search(str(txt or "")) for pat in COMPETITOR_PATTERNS.values())
    )
    return df


# ----------------------------------------------------------------------------
# FinBERT helpers: normalize pipeline output and fallback when only one label
# ----------------------------------------------------------------------------

def _finbert_to_scores(raw):
    """Normalize FinBERT pipeline output (list or single dict) to {label: score}."""
    if isinstance(raw, list):
        return {e["label"].lower(): e["score"] for e in raw}
    if isinstance(raw, dict):
        if "label" in raw and "score" in raw:
            return {raw["label"].lower(): raw["score"]}
        return {k.lower(): v for k, v in raw.items()}
    return None


def _parse_finbert_output(out):
    """Handle pipeline returning list of dicts or single dict; return {positive, neutral, negative} or partial."""
    if not out:
        return None
    if isinstance(out[0], dict) and "label" in out[0] and "score" in out[0]:
        return _finbert_to_scores(out)
    if isinstance(out[0], list):
        return _finbert_to_scores(out[0])
    return _finbert_to_scores(out[0])


def _finbert_scores_via_model(text: str):
    """When pipeline returns only one label, get all three scores from model logits (softmax)."""
    if not finbert or not torch or not getattr(finbert, "model", None):
        return None
    try:
        inputs = finbert.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = finbert.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        id2label = getattr(finbert.model.config, "id2label", None)
        if id2label is None:
            return None
        return {id2label[i].lower(): probs[i].item() for i in range(len(id2label))}
    except Exception:
        return None


def finbert_score(text: str) -> float | None:
    """Return sentiment (pos - neg) only when FinBERT is very confident neutral (>= FINBERT_THRESHOLD); else None for LLM fallback."""
    if not finbert:
        return None
    try:
        out = finbert(text)
    except RuntimeError as e:
        print(f"⚠️  FinBERT error (likely too long): {e}. Falling back to LLM.")
        return None
    except Exception as e:
        print(f"⚠️  FinBERT unexpected error: {e}. Falling back to LLM.")
        return None
    scores = _parse_finbert_output(out)
    # Pipeline sometimes returns only one label; then we need model logits to get all three scores
    if scores and len(scores) < 3:
        model_scores = _finbert_scores_via_model(text)
        if model_scores:
            scores = model_scores
    if not scores:
        return None
    pos, neu, neg = scores.get("positive", 0.0), scores.get("neutral", 0.0), scores.get("negative", 0.0)
    diff = pos - neg
    print(f"FinBERT → pos={pos:.2f}, neu={neu:.2f}, neg={neg:.2f}, diff={diff:.2f}")

    # Only trust FinBERT when it’s very sure it’s neutral
    if neu >= FINBERT_THRESHOLD:
        return diff
    # Otherwise caller (hybrid_sentiment) will use Gemini
    return None


PROMPT_SENTIMENT = """You are a sentiment analyst for financial forum comments about Questrade.

Task: Output only a JSON object with one key "sentiment" (float from -1 to 1, two decimal places). No other text.
- Use the full range for clear praise or clear criticism of Questrade based on the magnitude of the sentiment.
- Use smaller magnitude for ambiguous, sarcastic, or off-topic content.
- Comments that are longer and more detailed should have a higher magnitude of sentiment vs shorter quips or one-liners.

Examples:
"Been with Questrade 15 years, been great thank you Questrade" → {{"sentiment": 0.82}}
"Questrade is a money grab fees outrageous feature never works" → {{"sentiment": -0.62}}
"thanks youve been more helpful than the support team" (sarcastic) → {{"sentiment": -0.27}}
"Thank you Questrade was thinking of leaving but not anymore" → {{"sentiment": 0.43}}
"Is that stupid? Yes" (ambiguously negative not necessarily towards Questrade) → {{"sentiment": -0.10}}
"What a joke lol" (less ambiguously negative but still not necessarily towards Questrade) → {{"sentiment": -0.35}}
"us markets open time to buy the dip" (unrelated) → {{"sentiment": 0.04}}
"well iam impressed" (ambiguously positive not necessarily towards Questrade) → {{"sentiment": 0.15}}
"Thanks.   Setting it up now.   Much appreciated" (unrelated) → {{"sentiment": 0.10}}

Comment to rate:
\"\"\"
{text}
\"\"\"
"""


# ----------------------------------------------------------------------------
# Gemini LLM: general sentiment and competitor-specific prompt
# ----------------------------------------------------------------------------

def _llm_sentiment_one(prompt: str, text: str, label: str) -> float:
    """Call Gemini with prompt; parse JSON sentiment; retry on parse failure. Returns 0.0 on total failure."""
    for attempt in range(1, MAX_LLM_RETRIES + 1):
        print(f"LLM {label} attempt {attempt} →", text[:60].replace("\n", " "))
        resp = _llm_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        raw = (getattr(resp, "text", None) or "").strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                sent = float(json.loads(m.group())["sentiment"])
                print(f"LLM {label} → {sent:.2f}")
                return sent
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"⚠️  JSON parse error on attempt {attempt}: {e}. Raw: {raw!r}")
                continue
        print(f"⚠️  No JSON found on attempt {attempt}. Raw: {raw!r}")
    print(f"🚨 LLM {label} failed after {MAX_LLM_RETRIES} attempts.")
    return 0.0


def llm_sentiment(text: str) -> float:
    return _llm_sentiment_one(PROMPT_SENTIMENT.format(text=text), text, "eval")


PROMPT_COMPETITOR = """Comment mentions Questrade and competitors. Rate sentiment toward Questrade only: -1 (very negative) to +1 (very positive). Output only JSON: {{"sentiment": <float>}}, two decimal places.
- Comments that are longer and more detailed should have a higher magnitude of sentiment vs shorter quips or one-liners.

Examples:
"QT fees are worse than IBKR" → {{"sentiment": -0.61}}
"Moved from TD to Questrade, it's been great!" → {{"sentiment": 0.53}}
"Webull is fantastic!" → {{"sentiment": -0.32}}

Comment:
\"\"\"
{text}
\"\"\"
"""


def llm_competitor_sentiment(text: str) -> float:
    return _llm_sentiment_one(PROMPT_COMPETITOR.format(text=text), text, "competitor")


def hybrid_sentiment(row) -> float:
    """Implements the decision tree: competitor → Gemini (competitor prompt); else FinBERT if confident neutral, else Gemini (general)."""
    if row.competitors_mentioned:
        return llm_competitor_sentiment(row.clean_text)
    fb = finbert_score(row.clean_text)
    if fb is not None:
        return fb
    return llm_sentiment(row.clean_text)


def main():
    # Step 1: Load scraper CSV, drop junk, normalize text, add clean_text
    df = load_and_clean(DATA_PATH)
    # Step 2: Flag rows that mention competitors (these always use Gemini)
    df = tag_competitors(df)
    # Step 3: For each row, run the FinBERT vs Gemini decision tree and write score
    df["sentiment"] = df.apply(hybrid_sentiment, axis=1)
    out = Path(__file__).parent / OUTPUT_CSV
    df.to_csv(out, index=False)
    print(f"Saved sentiment output for {len(df)} rows to: {out}")


if __name__ == "__main__":
    main()
