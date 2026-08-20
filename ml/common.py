"""Shared loaders, body normalisation, vectorizer, and rule lists for the triage system."""
import hashlib
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = ROOT / "artifacts"  # given inputs, read-only
PROCESSED = DATA / "processed"

URGENCY_ORDER = ["routine", "same_day", "emergency"]  # ascending urgency


def _find(pattern):
    matches = list(RAW.glob(pattern))
    assert len(matches) == 1, f"expected one file matching {pattern}, got {matches}"
    return matches[0]


def load_messages():
    df = pd.read_csv(_find("*messages*.csv"))
    # DD/MM/YYYY here, ISO in holdout — parse explicitly per file, never infer.
    df["received_at"] = pd.to_datetime(df["received_at"], format="%d/%m/%Y %H:%M:%S")
    df = df.rename(columns={"urgency": "gold_label"})
    df["source"] = "messages"
    return df


def load_holdout():
    df = pd.read_csv(_find("*holdout*.csv"))
    df["received_at"] = pd.to_datetime(df["received_at"], format="%Y-%m-%d %H:%M")
    df["gold_label"] = pd.NA
    df["source"] = "holdout"
    return df


# --- body normalisation: strips the spliced-in noise, leaves the template core ---
_PHONE = re.compile(r"\b0\d[\d ]{8,11}\b")
_GREETING = re.compile(
    r"^(yeah hi|hi there|hello there|good (morning|afternoon|evening)"
    r"|hi|hiya|hello|hey|morning|afternoon|evening|erm)[,!. ]+", re.I)
_SIGNOFF = re.compile(
    r"(many thanks|thank you|thanks|cheers|kind regards|regards|best|ta)"
    r"[,!. ]*\w*[.!]?\s*$", re.I)


def normalise(body: str) -> str:
    b = str(body).lower()
    b = _PHONE.sub(" ", b)
    b = _GREETING.sub(" ", b)
    b = _SIGNOFF.sub(" ", b)
    b = re.sub(r"[^a-z0-9 ]", " ", b)
    return re.sub(r"\s+", " ", b).strip()


def core_hash(body: str) -> str:
    return hashlib.sha1(normalise(body).encode()).hexdigest()[:16]


def build_vectorizer():
    # word grams for meaning, char grams so 'boier'/'teh' typos still match
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
    ])

# Safety/de-escalation rules live in app/service.py — one copy, used by both channels.
