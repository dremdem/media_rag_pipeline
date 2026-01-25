"""SQLite database operations for Opinion Detector Service."""

import os
import sqlite3
from pathlib import Path
from typing import Any

# Allow override via environment variable for Docker flexibility
DB_PATH = Path(os.environ.get("OPINION_DB_PATH", "data/opinions.db"))


def _ensure_dirs() -> None:
    """Ensure database directory exists."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def init_db() -> None:
    """Initialize database schema."""
    _ensure_dirs()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS opinion_detection (
                chunk_id TEXT PRIMARY KEY,
                start REAL NOT NULL,
                end REAL NOT NULL,
                persons_json TEXT NOT NULL,
                has_opinion INTEGER NOT NULL,
                targets_json TEXT NOT NULL,
                spans_json TEXT NOT NULL,
                polarity TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.commit()


def upsert_detection(
    chunk_id: str,
    start: float,
    end: float,
    persons_json: str,
    has_opinion: int,
    targets_json: str,
    spans_json: str,
    polarity: str,
    confidence: float,
    created_at: str,
) -> None:
    """Insert or update detection result."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO opinion_detection
            (chunk_id, start, end, persons_json, has_opinion, targets_json, spans_json, polarity, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                start=excluded.start,
                end=excluded.end,
                persons_json=excluded.persons_json,
                has_opinion=excluded.has_opinion,
                targets_json=excluded.targets_json,
                spans_json=excluded.spans_json,
                polarity=excluded.polarity,
                confidence=excluded.confidence,
                created_at=excluded.created_at;
            """,
            (
                chunk_id,
                start,
                end,
                persons_json,
                has_opinion,
                targets_json,
                spans_json,
                polarity,
                confidence,
                created_at,
            ),
        )
        conn.commit()


def get_detection(chunk_id: str) -> dict[str, Any] | None:
    """Retrieve detection result by chunk_id."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            SELECT chunk_id, start, end, persons_json, has_opinion,
                   targets_json, spans_json, polarity, confidence, created_at
            FROM opinion_detection
            WHERE chunk_id = ?
            """,
            (chunk_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "chunk_id": row[0],
            "start": row[1],
            "end": row[2],
            "persons_json": row[3],
            "has_opinion": bool(row[4]),
            "targets_json": row[5],
            "spans_json": row[6],
            "polarity": row[7],
            "confidence": row[8],
            "created_at": row[9],
        }
