"""Transparent dictionary-based aspect evidence; no result when an aspect is absent."""
from preprocessing import clean_text

ASPECTS = {
    "Battery": ("battery", "charge", "charging", "பேட்டரி", "மின்கலம்"),
    "Camera": ("camera", "photo", "picture", "புகைப்படம்", "கேமரா"),
    "Price": ("price", "cost", "expensive", "cheap", "விலை"),
    "Quality": ("quality", "build", "durable", "தரம்"),
    "Delivery": ("delivery", "shipping", "arrived", "டெலிவரி"),
    "Customer service": ("support", "service", "customer care", "வாடிக்கையாளர் சேவை"),
}
POSITIVE = ("good", "great", "excellent", "love", "perfect", "fast", "நல்ல", "சிறப்பு", "அருமை")
NEGATIVE = ("bad", "poor", "broken", "terrible", "slow", "expensive", "overpriced", "เสีย", "மோசம்", "கெட்ட")

def analyze_aspects(text):
    cleaned = clean_text(text)
    clauses = __import__("re").split(r"[,.;!?]|\bbut\b|\bhowever\b", cleaned)
    rows = []
    for name, terms in ASPECTS.items():
        matches = [(clause, t) for clause in clauses for t in terms if clean_text(t) in clause]
        if not matches: continue
        # Use only the clause containing the aspect to reduce sentiment leakage across aspects.
        snippet, _ = matches[0]
        pos = any(clean_text(w) in snippet for w in POSITIVE)
        neg = any(clean_text(w) in snippet for w in NEGATIVE)
        label = "Positive" if pos and not neg else "Negative" if neg and not pos else "Insufficient evidence"
        rows.append({"Aspect": name, "Sentiment": label, "Evidence": snippet})
    return rows

