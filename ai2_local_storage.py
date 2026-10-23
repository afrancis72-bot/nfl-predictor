"""Local persistent upload storage independent of Streamlit UI.

Files survive normal restarts if the data directory is retained.
Replacements are archived transactionally. No silent deletion.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

def open_store(data_dir):
    directory = Path(data_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(directory / "slates.sqlite3", timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("""CREATE TABLE IF NOT EXISTS uploads (
        sport TEXT NOT NULL, kind TEXT NOT NULL, filename TEXT NOT NULL,
        content BLOB NOT NULL, uploaded_at TEXT NOT NULL,
        PRIMARY KEY(sport,kind))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS upload_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT, sport TEXT NOT NULL,
        kind TEXT NOT NULL, filename TEXT NOT NULL, content BLOB NOT NULL,
        uploaded_at TEXT NOT NULL, archived_at TEXT NOT NULL)""")
    return conn

def save_file(data_dir, sport, kind, filename, payload):
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("Nonempty file bytes required")
    if not filename or not sport or not kind:
        raise ValueError("File name, sport and kind required")
    now = datetime.now(timezone.utc).isoformat()
    with open_store(data_dir) as conn:
        old = conn.execute("SELECT filename,content,uploaded_at FROM uploads WHERE sport=? AND kind=?",
                           (sport, kind)).fetchone()
        if old:
            conn.execute("INSERT INTO upload_history(sport,kind,filename,content,uploaded_at,archived_at) VALUES(?,?,?,?,?,?)",
                         (sport,kind,*old,now))
        conn.execute("""INSERT INTO uploads(sport,kind,filename,content,uploaded_at)
                     VALUES(?,?,?,?,?) ON CONFLICT(sport,kind) DO UPDATE SET
                     filename=excluded.filename, content=excluded.content,
                     uploaded_at=excluded.uploaded_at""",
                     (sport,kind,filename,payload,now))

def load_file(data_dir, sport, kind):
    with open_store(data_dir) as conn:
        return conn.execute("SELECT filename,content,uploaded_at FROM uploads WHERE sport=? AND kind=?",
                            (sport,kind)).fetchone()

def list_history(data_dir, sport, kind, limit=20):
    with open_store(data_dir) as conn:
        return conn.execute("""SELECT id,filename,content,uploaded_at FROM upload_history
            WHERE sport=? AND kind=? ORDER BY id DESC LIMIT ?""",
            (sport,kind,limit)).fetchall()
