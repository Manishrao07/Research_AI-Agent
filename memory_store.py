"""
memory_store.py
SQLite-based persistence layer for ResearchAI Agent.

Har research session (topic, mode, report, metadata) ko save karta hai
taaki app restart hone ke baad bhi purani research dikh sake.
Har session ko ek chhota readable ID milta hai: rai-1, rai-2, rai-3...
"""

import sqlite3
import json
import os
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.environ.get("SQLITE_DB_PATH", "research_history.db")


@contextmanager
def get_connection():
    """Context manager taaki connection hamesha properly close ho, even on error."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # rows ko dict-like access ke liye
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Database table banata hai agar exist nahi karta."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS research_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                topic TEXT NOT NULL,
                mode TEXT NOT NULL,
                report TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
        """)


def save_session(topic: str, mode: str, report: str, metadata: dict = None) -> str:
    """
    Ek research session save karta hai, chhota readable session_id generate karke (rai-1, rai-2...).
    mode: 'quick' ya 'deep'
    metadata: extra info jaise confidence score, critic verdict, sub_questions (dict, JSON ban jayega)
    Returns: generated session_id (string, jaise "rai-7")
    """
    metadata_json = json.dumps(metadata or {})
    timestamp = datetime.now().isoformat()

    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO research_sessions (session_id, topic, mode, report, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("", topic, mode, report, metadata_json, timestamp)
        )
        row_id = cursor.lastrowid
        session_id = f"rai-{row_id}"
        conn.execute(
            "UPDATE research_sessions SET session_id = ? WHERE id = ?",
            (session_id, row_id)
        )
        return session_id


def get_recent_sessions(limit: int = 10) -> list[dict]:
    """Sabse recent research sessions list karta hai (sirf summary, poora report nahi)."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT session_id, topic, mode, created_at FROM research_sessions ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


def get_session_by_session_id(session_id: str) -> dict:
    """Ek specific session ka poora data (report + metadata) deta hai, session_id se (jaise 'rai-7')."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM research_sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        if row:
            data = dict(row)
            data["metadata"] = json.loads(data["metadata"]) if data["metadata"] else {}
            return data
        return None


def delete_session(session_id: str):
    """Ek session delete karta hai, session_id se."""
    with get_connection() as conn:
        conn.execute("DELETE FROM research_sessions WHERE session_id = ?", (session_id,))


# Module load hote hi DB initialize ho jaye
init_db()


if __name__ == "__main__":
    print("Testing memory_store.py...\n")

    test_session_id = save_session(
        topic="Test topic: AI in healthcare",
        mode="quick",
        report="## Test Report\nThis is a test report content.",
        metadata={"confidence": {"score": 85, "label": "High"}}
    )
    print(f"Saved session with id: {test_session_id}")

    test_session_id_2 = save_session(
        topic="Test topic 2: EV market trends",
        mode="deep",
        report="## Test Report 2\nAnother test report.",
        metadata={"critic_verdict": {"verdict": "PASS"}}
    )
    print(f"Saved session with id: {test_session_id_2}")

    recent = get_recent_sessions(limit=5)
    print(f"\nRecent sessions ({len(recent)}):")
    for s in recent:
        print(f"  [{s['session_id']}] {s['topic']} ({s['mode']}) — {s['created_at']}")

    full = get_session_by_session_id(test_session_id)
    print(f"\nFull session {test_session_id}:")
    print(f"  Topic: {full['topic']}")
    print(f"  Report length: {len(full['report'])} chars")
    print(f"  Metadata: {full['metadata']}")