"""Train and compare sentiment classifiers on dataset.csv."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from preprocessing import clean_text
from config import settings

BASE = settings.project_dir

def find_column(columns, candidates):
    lookup = {str(c).strip().lower(): c for c in columns}
    return next((lookup[c] for c in candidates if c in lookup), None)

def train(csv_path=None, output_dir=None):
    csv_path = Path(csv_path) if csv_path else settings.dataset_path
    output_dir = Path(output_dir) if output_dir else settings.model_dir
    df = pd.read_csv(csv_path)
    text_col = find_column(df.columns, ("review_text", "review", "text", "content"))
    label_col = find_column(df.columns, ("sentiment", "label", "target"))
    if not text_col or not label_col:
        raise ValueError("CSV needs a review text column (review/review_text/text) and a sentiment label column.")
    before = len(df)
    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip().str.title()
    df = df[(df[text_col] != "") & (df[label_col] != "")].drop_duplicates(subset=[text_col, label_col])
    df["cleaned"] = df[text_col].map(clean_text)
    df = df[df.cleaned != ""]
    if df[label_col].nunique() < 2 or df[label_col].value_counts().min() < 5:
        raise ValueError("Need at least two sentiment classes and five examples per class for a stratified split.")
    y = df[label_col]
    X_train, X_hold, y_train, y_hold = train_test_split(df.cleaned, y, test_size=settings.holdout_fraction, random_state=settings.random_state, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_hold, y_hold, test_size=settings.test_fraction_of_holdout, random_state=settings.random_state, stratify=y_hold)
    constructors = {
        "Logistic Regression (calibrated)": lambda: CalibratedClassifierCV(LogisticRegression(max_iter=settings.model_max_iter, class_weight="balanced", random_state=settings.random_state), cv=3),
        "Multinomial Naive Bayes (calibrated)": lambda: CalibratedClassifierCV(MultinomialNB(), cv=3),
        "Linear SVM (calibrated)": lambda: CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=settings.random_state), cv=3),
    }
    selection = {}
    for name, model_factory in constructors.items():
        pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(settings.tfidf_ngram_min,settings.tfidf_ngram_max), max_features=settings.tfidf_max_features, sublinear_tf=True)), ("model", model_factory())])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        selection[name] = {"validation_f1_weighted": f1_score(y_val, pred, average="weighted", zero_division=0), "factory": model_factory}
        print(f"{name}: validation weighted F1={selection[name]['validation_f1_weighted']:.4f}")
    winner = max(selection, key=lambda n: selection[n]["validation_f1_weighted"])
    # Refit the selected model on train+validation; test remains held out for final evaluation.
    final = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(settings.tfidf_ngram_min,settings.tfidf_ngram_max), max_features=settings.tfidf_max_features, sublinear_tf=True)), ("model", selection[winner]["factory"]())])
    final.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    pred = final.predict(X_test)
    labels = list(final.classes_)
    metrics = {
        "selected_model": winner,
        "validation_scores": {n: v["validation_f1_weighted"] for n,v in selection.items()},
        "test": {"accuracy": accuracy_score(y_test,pred), "precision_weighted": precision_score(y_test,pred,average="weighted",zero_division=0), "recall_weighted": recall_score(y_test,pred,average="weighted",zero_division=0), "f1_weighted": f1_score(y_test,pred,average="weighted",zero_division=0), "report": classification_report(y_test,pred,output_dict=True,zero_division=0), "confusion_matrix": confusion_matrix(y_test,pred,labels=labels).tolist(), "labels": labels},
        "dataset": {"rows_used": len(df), "rows_removed": before-len(df), "class_counts": y.value_counts().to_dict(), "text_column": str(text_col), "label_column": str(label_col)},
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "metrics": metrics}, output_dir / "sentiment_bundle.joblib")
    # Keep legacy artifact names available for older project versions.
    joblib.dump(final.named_steps["model"], output_dir / "sentiment_model.pkl")
    joblib.dump(final.named_steps["tfidf"], output_dir / "tfidf_vectorizer.pkl")
    print(f"Selected: {winner}; held-out test metrics: {metrics['test']}")
    print(f"Dataset class counts: {metrics['dataset']['class_counts']}; rows used={len(df)}, removed={before-len(df)}")
    print("Saved sentiment_bundle.joblib and legacy model/vectorizer artifacts.")
    return metrics

if __name__ == "__main__":
    train()
