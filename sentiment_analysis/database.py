"""SQLite persistence for analyzed reviews."""
import sqlite3
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from config import settings

DB_PATH = settings.history_db_path

@contextmanager
def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

def initialize():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT, review_text TEXT NOT NULL,
            sentiment TEXT, rating INTEGER, confidence REAL, suspicious_score REAL,
            created_at TEXT NOT NULL, source TEXT DEFAULT 'single')""")

def add_review(review_text, sentiment=None, rating=None, confidence=None, suspicious_score=None, source="single"):
    with connect() as con:
        cur = con.execute("INSERT INTO reviews(review_text,sentiment,rating,confidence,suspicious_score,created_at,source) VALUES(?,?,?,?,?,?,?)",
            (review_text, sentiment, rating, confidence, suspicious_score, datetime.now().astimezone().isoformat(timespec="seconds"), source))
        return cur.lastrowid

def history(sentiment=None, start_date=None, end_date=None, search=""):
    sql = "SELECT * FROM reviews WHERE 1=1"
    args = []
    if sentiment and sentiment != "All": sql += " AND sentiment=?"; args.append(sentiment)
    if start_date: sql += " AND date(created_at)>=date(?)"; args.append(str(start_date))
    if end_date: sql += " AND date(created_at)<=date(?)"; args.append(str(end_date))
    if search: sql += " AND review_text LIKE ?"; args.append(f"%{search}%")
    sql += " ORDER BY created_at DESC"
    with connect() as con: return [dict(r) for r in con.execute(sql, args).fetchall()]

def delete_review(record_id):
    with connect() as con: con.execute("DELETE FROM reviews WHERE id=?", (int(record_id),))

