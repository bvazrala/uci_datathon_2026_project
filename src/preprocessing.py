"""Text preprocessing utilities."""
import re


def clean_tweet(text: str) -> str:
    """Basic tweet cleaning: lowercase, strip URLs, mentions, hashtags."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
