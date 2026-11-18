"""Single and bulk prediction helpers shared by the Streamlit app."""
from preprocessing import clean_text

def predict_one(text, bundle):
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Review text cannot be empty.")
    model = bundle["model"]
    if "legacy_vectorizer" in bundle:
        vector = bundle["legacy_vectorizer"].transform([cleaned])
        label = model.predict(vector)[0]
        probs = model.predict_proba(vector)[0] if hasattr(model, "predict_proba") else None
    else:
        label = model.predict([cleaned])[0]
        probs = model.predict_proba([cleaned])[0] if hasattr(model, "predict_proba") else None
    return str(label), (float(max(probs)) if probs is not None else None), ({str(c): float(p) for c,p in zip(model.classes_, probs)} if probs is not None else {})

def predict_many(texts, bundle):
    output = []
    for text in texts:
        if text is None or not str(text).strip():
            output.append({"predicted_sentiment": None, "confidence": None, "probabilities": {}, "error": "Review text is empty."})
            continue
        label, confidence, probabilities = predict_one(text, bundle)
        output.append({"predicted_sentiment": label, "confidence": confidence, "probabilities": probabilities, "error": None})
    return output

