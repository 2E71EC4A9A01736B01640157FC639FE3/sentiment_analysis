"""Shared text preprocessing for training and inference."""
import re
import unicodedata

NEGATIONS = {"not", "no", "never", "n't"}

def clean_text(value):
    """Normalize whitespace and casing while preserving Tamil, emoji and negation."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = re.sub(r"(\w)\1{2,}", r"\1\1", text, flags=re.UNICODE)
    # Keep letters/digits from all scripts, apostrophes (negation contractions), and emoji.
    text = "".join(ch if (ch.isalnum() or ch.isspace() or unicodedata.category(ch).startswith("M") or ch in "'’" or ord(ch) > 0x1F000) else " " for ch in text)
    return re.sub(r"\s+", " ", text).strip()

