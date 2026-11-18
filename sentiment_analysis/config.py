"""Central application configuration loaded from environment and project .env."""
from dataclasses import dataclass
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

def _load_dotenv(path):
    """Load simple KEY=VALUE entries without overriding real environment variables."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)

def _path_setting(name, default):
    path = Path(os.getenv(name, default)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()

def _int_setting(name, default, minimum=1):
    try:
        result = int(os.getenv(name, str(default)))
        if result < minimum:
            raise ValueError
        return result
    except ValueError:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")

def _float_setting(name, default, minimum=0.01, maximum=0.99):
    try:
        result = float(os.getenv(name, str(default)))
        if not minimum <= result <= maximum:
            raise ValueError
        return result
    except ValueError:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")

@dataclass(frozen=True)
class Settings:
    project_dir: Path
    app_title: str
    app_icon: str
    app_layout: str
    dataset_path: Path
    model_dir: Path
    history_db_path: Path
    tfidf_max_features: int
    tfidf_ngram_min: int
    tfidf_ngram_max: int
    model_max_iter: int
    random_state: int
    holdout_fraction: float
    test_fraction_of_holdout: float
    max_upload_mb: int

    @property
    def sentiment_bundle_path(self): return self.model_dir / "sentiment_bundle.joblib"
    @property
    def sentiment_model_path(self): return self.model_dir / "sentiment_model.pkl"
    @property
    def vectorizer_path(self): return self.model_dir / "tfidf_vectorizer.pkl"
    @property
    def rating_model_path(self): return self.model_dir / "rating_model.joblib"

def load_settings():
    _load_dotenv(PROJECT_DIR / ".env")
    ngram_min = _int_setting("TFIDF_NGRAM_MIN", 1)
    ngram_max = _int_setting("TFIDF_NGRAM_MAX", 2)
    if ngram_max < ngram_min:
        raise ValueError("TFIDF_NGRAM_MAX must be greater than or equal to TFIDF_NGRAM_MIN.")
    try:
        random_state = int(os.getenv("RANDOM_STATE", "42"))
    except ValueError as exc:
        raise ValueError("RANDOM_STATE must be an integer.") from exc
    layout = os.getenv("APP_LAYOUT", "wide").strip().lower()
    if layout not in {"wide", "centered"}:
        raise ValueError("APP_LAYOUT must be either 'wide' or 'centered'.")
    return Settings(
        project_dir=PROJECT_DIR,
        app_title=os.getenv("APP_TITLE", "Product Review Sentiment Studio"),
        app_icon=os.getenv("APP_ICON", "🛍️"),
        app_layout=layout,
        dataset_path=_path_setting("DATASET_PATH", "dataset.csv"),
        model_dir=_path_setting("MODEL_DIR", "."),
        history_db_path=_path_setting("HISTORY_DB_PATH", "review_history.db"),
        tfidf_max_features=_int_setting("TFIDF_MAX_FEATURES", 12000),
        tfidf_ngram_min=ngram_min,
        tfidf_ngram_max=ngram_max,
        model_max_iter=_int_setting("MODEL_MAX_ITER", 1500),
        random_state=random_state,
        holdout_fraction=_float_setting("HOLDOUT_FRACTION", .40),
        test_fraction_of_holdout=_float_setting("TEST_FRACTION_OF_HOLDOUT", .50),
        max_upload_mb=_int_setting("MAX_UPLOAD_MB", 20),
    )

settings = load_settings()
