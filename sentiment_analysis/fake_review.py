"""Heuristic suspiciousness indicators. These do not establish that a review is fake."""
import re
from preprocessing import clean_text

def analyze_suspiciousness(text, known_texts=()):
    cleaned = clean_text(text)
    flags = []
    if len(cleaned) >= 12 and cleaned in {clean_text(x) for x in known_texts if x}:
        flags.append("Exact or near duplicate text")
    words = cleaned.split()
    if len(words) >= 8 and len(set(words)) / len(words) < 0.45:
        flags.append("Unusually repetitive vocabulary")
    if re.search(r"(.)\1{4,}", str(text), flags=re.UNICODE):
        flags.append("Long repeated character sequence")
    if len(words) >= 5 and max((words.count(w) for w in set(words)), default=0) / len(words) > 0.35:
        flags.append("One word occurs unusually often")
    return {"suspicious_score": min(100, len(flags) * 25), "suspicious_warning": "; ".join(flags) if flags else "No heuristic warning"}

