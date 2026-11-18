"""Optional rating model. Requires genuine numeric rating labels in the source data."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import clean_text
from config import settings

def train_rating_model(csv_path=None, output_path=None):
    csv_path = csv_path or settings.dataset_path
    output_path = output_path or settings.rating_model_path
    df = pd.read_csv(csv_path)
    col = next((c for c in df.columns if c.lower() in {"rating", "stars", "star_rating"}), None)
    text_col = next((c for c in df.columns if c.lower() in {"review", "review_text", "text"}), None)
    if not col or not text_col: raise ValueError("Rating training requires review text and actual rating labels (1–5).")
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col, text_col]); df = df[df[col].between(1, 5)]
    if df[col].nunique() < 2 or df[col].value_counts().min() < 2: raise ValueError("At least two rating classes with two examples each are required.")
    Xtr, Xte, ytr, yte = train_test_split(df[text_col].map(clean_text), df[col].astype(int), test_size=.2, random_state=42, stratify=df[col])
    pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(settings.tfidf_ngram_min, settings.tfidf_ngram_max), max_features=settings.tfidf_max_features)), ("model", LogisticRegression(max_iter=settings.model_max_iter, class_weight="balanced", random_state=settings.random_state))])
    pipe.fit(Xtr, ytr)
    print(classification_report(yte, pipe.predict(Xte), zero_division=0))
    joblib.dump(pipe, output_path)
    return pipe

if __name__ == "__main__":
    train_rating_model()

