"""Tweet preprocessing — mirrors the notebook cleaning pipeline."""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

CUSTOM_STOPS = list(ENGLISH_STOP_WORDS.union({
    "amp",        # HTML &amp; artifact
    "starbucks",  # brand name uniform across classes
    "walmart",
    "mcdonalds",
    "dominos",
    "subway",
    "rt",         # retweet marker
    "got", "get", "just", "like", "bet",
}))


def clean_tweet(text: str) -> str:
    """
    Cleaning pipeline (matches notebook):
    1. Lowercase
    2. Remove RT prefix
    3. Remove URLs
    4. Remove @mentions
    5. Remove # symbol (keep hashtag word)
    6. Remove special chars / numbers — keep letters and ?!'
    7. Strip extra whitespace
    """
    text = text.lower()
    text = re.sub(r"^rt\s+", "", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s!?']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
