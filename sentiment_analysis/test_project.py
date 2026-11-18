"""Small regression tests for project helpers. Run: python -m unittest -v"""
import tempfile
import unittest
from pathlib import Path
import database
from preprocessing import clean_text
from aspect_analysis import analyze_aspects
from fake_review import analyze_suspiciousness
from reports import csv_bytes, excel_bytes, chart_png_bytes
from train_model import find_column
from rating_model import train_rating_model
from prediction import predict_many
from train_model import train
from config import load_settings
from unittest.mock import patch

class ProjectTests(unittest.TestCase):
    def test_config_environment_overrides_env_file_and_resolves_paths(self):
        with patch.dict("os.environ", {"MAX_UPLOAD_MB": "7", "MODEL_DIR": "custom-models"}, clear=False):
            configured = load_settings()
        self.assertEqual(configured.max_upload_mb, 7)
        self.assertTrue(str(configured.model_dir).endswith("custom-models"))

    def test_preprocessing_keeps_negation_tamil_and_normalizes(self):
        text = clean_text("NOT good!!!  soooo 😊 நல்லது")
        self.assertIn("not", text); self.assertIn("so", text); self.assertIn("நல்லது", text)

    def test_aspect_only_when_mentioned_and_sentiment_evidence_present(self):
        rows = analyze_aspects("The battery is great, but price is expensive.")
        got = {r["Aspect"]: r["Sentiment"] for r in rows}
        self.assertEqual(got["Battery"], "Positive")
        self.assertEqual(got["Price"], "Negative")
        self.assertEqual(analyze_aspects("Everything is okay."), [])

    def test_suspiciousness_is_a_warning_not_a_verdict(self):
        result = analyze_suspiciousness("same review text", ["same review text"])
        self.assertGreaterEqual(result["suspicious_score"], 0)
        self.assertIn("suspicious_warning", result)

    def test_column_aliases(self):
        self.assertEqual(find_column(["Review_Text", "sentiment"], ("review_text", "review")), "Review_Text")

    def test_database_add_filter_and_delete(self):
        with tempfile.TemporaryDirectory() as folder:
            previous = database.DB_PATH
            database.DB_PATH = Path(folder) / "history.sqlite"
            try:
                database.initialize()
                row_id = database.add_review("not good", "Negative", 1, .8, 25)
                found = database.history("Negative", search="not")
                self.assertEqual(found[0]["id"], row_id)
                database.delete_review(row_id)
                self.assertEqual(database.history(), [])
            finally: database.DB_PATH = previous

    def test_csv_and_excel_exports(self):
        rows = [{"review_text": "great", "predicted_sentiment": "Positive", "confidence": .9}]
        self.assertIn(b"predicted_sentiment", csv_bytes(rows))
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            self.skipTest("openpyxl is listed in requirements.txt but is not installed in this environment")
        self.assertTrue(excel_bytes(rows).startswith(b"PK"))

    def test_sentiment_chart_png_downloads(self):
        import pandas as pd
        counts = pd.Series([4, 2, 3], index=["Positive", "Neutral", "Negative"])
        self.assertTrue(chart_png_bytes(counts, "bar").startswith(b"\x89PNG"))
        self.assertTrue(chart_png_bytes(counts, "pie").startswith(b"\x89PNG"))

    def test_rating_training_refuses_missing_ground_truth(self):
        with tempfile.TemporaryDirectory() as folder:
            csv = Path(folder) / "reviews.csv"
            csv.write_text("review,sentiment\ngood,Positive\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "actual rating labels"):
                train_rating_model(csv)

    def test_rating_training_with_real_numeric_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            csv = Path(folder) / "ratings.csv"
            lines = ["review_text,rating"]
            for rating, word in enumerate(("awful", "poor", "okay", "nice", "excellent"), 1):
                for i in range(8): lines.append(f'"{word} product sample {i}",{rating}')
            csv.write_text("\n".join(lines), encoding="utf-8")
            model = train_rating_model(csv, Path(folder) / "rating.joblib")
            self.assertIn(int(model.predict([clean_text("excellent product")])[0]), range(1, 6))

    def test_bulk_prediction_returns_each_result_and_handles_empty(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        texts = ["very good", "very bad", "very good product", "very bad product"]
        labels = ["Positive", "Negative", "Positive", "Negative"]
        model = Pipeline([("tfidf", TfidfVectorizer()), ("model", LogisticRegression())]).fit(texts, labels)
        rows = predict_many(["very good", "  "], {"model": model})
        self.assertEqual(rows[0]["predicted_sentiment"], "Positive")
        self.assertIsNotNone(rows[0]["confidence"])
        self.assertIsNotNone(rows[1]["error"])

    def test_training_comparison_metrics_and_missing_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            csv = Path(folder) / "sentiment.csv"
            lines = ["review,sentiment"]
            for label, word, size in (("Positive", "good", 40), ("Negative", "bad", 20), ("Neutral", "ordinary", 10)):
                for i in range(size): lines.append(f'"{word} sample review {i}",{label}')
            csv.write_text("\n".join(lines), encoding="utf-8")
            metrics = train(csv, Path(folder) / "artifacts")
            self.assertEqual(len(metrics["validation_scores"]), 3)
            self.assertIn("confusion_matrix", metrics["test"])
            self.assertEqual(metrics["dataset"]["class_counts"]["Positive"], 40)
            bad = Path(folder) / "bad.csv"
            bad.write_text("title\nhello\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "review text column"):
                train(bad, Path(folder) / "bad_artifacts")

if __name__ == "__main__": unittest.main()
