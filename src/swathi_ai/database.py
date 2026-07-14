from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class ChatRepository:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, check_same_thread=False)

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_session
                ON conversations(session_id, id)
                """
            )
            conn.commit()

    def save(self, session_id: str, role: str, message: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations(session_id, role, message, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    message,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def load(self, session_id: str) -> list[tuple[str, str]]:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT role, message
                FROM conversations
                WHERE session_id = ?
                ORDER BY id
                """,
                (session_id,),
            ).fetchall()

    def sessions(self) -> list[tuple[str, str, int]]:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT session_id, MIN(created_at), COUNT(*)
                FROM conversations
                GROUP BY session_id
                ORDER BY MIN(created_at) DESC
                """
            ).fetchall()

    def delete(self, session_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM conversations WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()

    def clear(self) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM conversations")
            conn.commit()
