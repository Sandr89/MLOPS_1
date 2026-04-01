"""
Text preprocessing module for Twitter Hate Speech dataset.
"""

import re
import string


def clean_tweet(text: str) -> str:
    """
    Clean tweet text: remove URLs, @mentions, emoji, punctuation.
    """
    if not isinstance(text, str) or not text:
        return ""

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbol but keep the word
    text = re.sub(r"#", "", text)

    # Remove emoji and special unicode
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    # Lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_text_column(df):
    """Get tweet text column (handles different naming conventions)."""
    for col in ["tweet", "Tweet", "text", "Text", "tweet_text"]:
        if col in df.columns:
            return col
    raise ValueError(f"Text column not found. Available: {list(df.columns)}")


def get_label_column(df):
    """Get label column."""
    if "label" in df.columns:
        return "label"
    if "Label" in df.columns:
        return "Label"
    raise ValueError(f"Label column not found. Available: {list(df.columns)}")
