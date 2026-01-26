"""SQLite database operations for LLM Analyzer Service."""

import os
import sqlite3
from pathlib import Path
from typing import Any

# Allow override via environment variable for Docker flexibility
DB_PATH = Path(os.environ.get("LLM_ANALYZER_DB_PATH", "data/analyzer.db"))


def _ensure_dirs() -> None:
    """Ensure database directory exists."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def init_db() -> None:
    """Initialize database schema."""
    _ensure_dirs()
    with sqlite3.connect(DB_PATH) as conn:
        # Opinion detection table
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

        # Q&A boundary segments table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS qa_boundary (
                video_id TEXT NOT NULL,
                seg_id TEXT NOT NULL,
                type TEXT NOT NULL,
                start_u INTEGER NOT NULL,
                end_u INTEGER NOT NULL,
                start REAL NOT NULL,
                end REAL NOT NULL,
                confidence REAL NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (video_id, seg_id)
            );
            """
        )

        # Q&A semantic blocks table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS qa_block (
                video_id TEXT NOT NULL,
                block_id TEXT NOT NULL,
                start_u INTEGER NOT NULL,
                end_u INTEGER NOT NULL,
                start REAL NOT NULL,
                end REAL NOT NULL,
                questions_json TEXT NOT NULL,
                answer_summary TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (video_id, block_id)
            );
            """
        )

        # Q&A export cache table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS qa_export (
                video_id TEXT PRIMARY KEY,
                export_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )

        conn.commit()


# --- Opinion Detection Operations ---


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


# --- Q&A Boundary Operations ---


def upsert_boundary(
    video_id: str,
    seg_id: str,
    seg_type: str,
    start_u: int,
    end_u: int,
    start: float,
    end: float,
    confidence: float,
    notes: str | None,
    created_at: str,
) -> None:
    """Insert or update Q&A boundary segment."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO qa_boundary
            (video_id, seg_id, type, start_u, end_u, start, end, confidence, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id, seg_id) DO UPDATE SET
                type=excluded.type,
                start_u=excluded.start_u,
                end_u=excluded.end_u,
                start=excluded.start,
                end=excluded.end,
                confidence=excluded.confidence,
                notes=excluded.notes,
                created_at=excluded.created_at;
            """,
            (video_id, seg_id, seg_type, start_u, end_u, start, end, confidence, notes, created_at),
        )
        conn.commit()


def get_boundaries(video_id: str) -> list[dict[str, Any]]:
    """Retrieve all boundary segments for a video."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            SELECT video_id, seg_id, type, start_u, end_u, start, end, confidence, notes, created_at
            FROM qa_boundary
            WHERE video_id = ?
            ORDER BY start_u
            """,
            (video_id,),
        )
        rows = cur.fetchall()
        return [
            {
                "video_id": row[0],
                "seg_id": row[1],
                "type": row[2],
                "start_u": row[3],
                "end_u": row[4],
                "start": row[5],
                "end": row[6],
                "confidence": row[7],
                "notes": row[8],
                "created_at": row[9],
            }
            for row in rows
        ]


def delete_boundaries(video_id: str) -> None:
    """Delete all boundary segments for a video (for re-processing)."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM qa_boundary WHERE video_id = ?", (video_id,))
        conn.commit()


# --- Q&A Block Operations ---


def upsert_block(
    video_id: str,
    block_id: str,
    start_u: int,
    end_u: int,
    start: float,
    end: float,
    questions_json: str,
    answer_summary: str,
    confidence: float,
    created_at: str,
) -> None:
    """Insert or update Q&A semantic block."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO qa_block
            (video_id, block_id, start_u, end_u, start, end, questions_json, answer_summary, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id, block_id) DO UPDATE SET
                start_u=excluded.start_u,
                end_u=excluded.end_u,
                start=excluded.start,
                end=excluded.end,
                questions_json=excluded.questions_json,
                answer_summary=excluded.answer_summary,
                confidence=excluded.confidence,
                created_at=excluded.created_at;
            """,
            (video_id, block_id, start_u, end_u, start, end, questions_json, answer_summary, confidence, created_at),
        )
        conn.commit()


def get_blocks(video_id: str) -> list[dict[str, Any]]:
    """Retrieve all Q&A blocks for a video."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            SELECT video_id, block_id, start_u, end_u, start, end, questions_json, answer_summary, confidence, created_at
            FROM qa_block
            WHERE video_id = ?
            ORDER BY start_u
            """,
            (video_id,),
        )
        rows = cur.fetchall()
        return [
            {
                "video_id": row[0],
                "block_id": row[1],
                "start_u": row[2],
                "end_u": row[3],
                "start": row[4],
                "end": row[5],
                "questions_json": row[6],
                "answer_summary": row[7],
                "confidence": row[8],
                "created_at": row[9],
            }
            for row in rows
        ]


def delete_blocks(video_id: str) -> None:
    """Delete all Q&A blocks for a video (for re-processing)."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM qa_block WHERE video_id = ?", (video_id,))
        conn.commit()


# --- Q&A Export Operations ---


def upsert_export(video_id: str, export_json: str, created_at: str) -> None:
    """Insert or update Q&A export."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO qa_export (video_id, export_json, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                export_json=excluded.export_json,
                created_at=excluded.created_at;
            """,
            (video_id, export_json, created_at),
        )
        conn.commit()


def get_export(video_id: str) -> dict[str, Any] | None:
    """Retrieve Q&A export for a video."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT video_id, export_json, created_at FROM qa_export WHERE video_id = ?",
            (video_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "video_id": row[0],
            "export_json": row[1],
            "created_at": row[2],
        }
