"""Persistent, sport-scoped settings for TEAM AI² Pro.

The same AI2_DATA_DIR durable volume used by unified_app.py stores this DB.
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

VALID_SPORTS = frozenset({"NFL", "PGA", "NASCAR", "NBA"})

def initialize_settings(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS sport_settings (
        sport TEXT NOT NULL,
        setting_key TEXT NOT NULL,
        value_json TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (sport, setting_key)
    )""")

def save_setting(db_path: Path, sport: str, key: str, value) -> None:
    if sport not in VALID_SPORTS:
        raise ValueError("Unknown sport")
    if not key or len(key) > 128:
        raise ValueError("Invalid setting key")
    encoded = json.dumps(value, allow_nan=False)
    with sqlite3.connect(db_path, timeout=10) as conn:
        initialize_settings(conn)
        conn.execute("""INSERT INTO sport_settings
            (sport,setting_key,value_json,updated_at) VALUES(?,?,?,?)
            ON CONFLICT(sport,setting_key) DO UPDATE SET
            value_json=excluded.value_json,updated_at=excluded.updated_at""",
            (sport,key,encoded,datetime.now(timezone.utc).isoformat()))

def load_setting(db_path: Path, sport: str, key: str, default=None):
    if sport not in VALID_SPORTS:
        raise ValueError("Unknown sport")
    with sqlite3.connect(db_path, timeout=10) as conn:
        initialize_settings(conn)
        row = conn.execute(
            "SELECT value_json FROM sport_settings WHERE sport=? AND setting_key=?",
            (sport,key)).fetchone()
    return json.loads(row[0]) if row else default
