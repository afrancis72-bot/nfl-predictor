"""Persist NFL candidate review portfolios without mutating saved input slates."""
import hashlib
import json
from datetime import datetime, timezone
from ai2_local_storage import open_store

def slate_fingerprint(salaries, projections):
    if not salaries or not projections:
        raise ValueError("Both salary and projection files are required")
    h = hashlib.sha256()
    for record in (salaries, projections):
        content = record[1]
        h.update(len(content).to_bytes(8, "big"))
        h.update(content)
    return h.hexdigest()

def save_portfolio(data_dir, fingerprint, candidates):
    if not fingerprint or not candidates:
        raise ValueError("Slate fingerprint and candidates required")
    payload = json.dumps(candidates, sort_keys=True)
    with open_store(data_dir) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS nfl_portfolios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slate_fingerprint TEXT NOT NULL,
            created_at TEXT NOT NULL,
            payload TEXT NOT NULL)""")
        cursor = conn.execute(
            "INSERT INTO nfl_portfolios(slate_fingerprint,created_at,payload) VALUES(?,?,?)",
            (fingerprint, datetime.now(timezone.utc).isoformat(), payload))
        return cursor.lastrowid

def load_latest_portfolio(data_dir, fingerprint):
    with open_store(data_dir) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS nfl_portfolios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slate_fingerprint TEXT NOT NULL,
            created_at TEXT NOT NULL,
            payload TEXT NOT NULL)""")
        row = conn.execute("""SELECT id,created_at,payload FROM nfl_portfolios
            WHERE slate_fingerprint=? ORDER BY id DESC LIMIT 1""",
            (fingerprint,)).fetchone()
    if row is None:
        return None
    return {"id": row[0], "created_at": row[1], "candidates": json.loads(row[2])}
