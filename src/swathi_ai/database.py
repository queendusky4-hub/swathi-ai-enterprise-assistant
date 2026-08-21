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
        connection = sqlite3.connect(
           self.path,
           timeout=60.0,
           check_same_thread=False,
    )

        connection.execute("PRAGMA busy_timeout = 60000")
        connection.row_factory = sqlite3.Row
        return connection

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL
                        CHECK(role IN ('user', 'assistant')),
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_conversations_session
                ON conversations(session_id, id)
                """
            )

            conn.commit()

    def save(
        self,
        session_id: str,
        role: str,
        message: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations(
                    session_id,
                    role,
                    message,
                    created_at
                )
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

    def load(
        self,
        session_id: str,
    ) -> list[tuple[str, str]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT role, message
                FROM conversations
                WHERE session_id = ?
                ORDER BY id
                """,
                (session_id,),
            ).fetchall()

        return [
            (row["role"], row["message"])
            for row in rows
        ]

    def sessions(self) -> list[tuple[str, str, int]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    session_id,
                    MIN(created_at) AS first_message,
                    COUNT(*) AS message_count
                FROM conversations
                GROUP BY session_id
                ORDER BY first_message DESC
                """
            ).fetchall()

        return [
            (
                row["session_id"],
                row["first_message"],
                row["message_count"],
            )
            for row in rows
        ]

    def delete(self, session_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                DELETE FROM conversations
                WHERE session_id = ?
                """,
                (session_id,),
            )
            conn.commit()

    def clear(self) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM conversations")
            conn.commit()


 

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_tokens (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id)
                        REFERENCES users(id)
                        ON DELETE CASCADE
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_auth_tokens_user
                ON auth_tokens(user_id)
                """
            )

            conn.commit()

    def create_user(
        self,
        username: str,
        password_hash: str,
        password_salt: str,
    ) -> dict:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users(
                    username,
                    password_hash,
                    password_salt,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    username,
                    password_hash,
                    password_salt,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            conn.commit()

            user_id = cursor.lastrowid

        return {
            "id": user_id,
            "username": username,
        }

    def get_user_by_username(
        self,
        username: str,
    ) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    id,
                    username,
                    password_hash,
                    password_salt,
                    created_at
                FROM users
                WHERE username = ?
                """,
                (username,),
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    def save_token(
        self,
        token: str,
        user_id: int,
        expires_at: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO auth_tokens(
                    token,
                    user_id,
                    expires_at,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    token,
                    user_id,
                    expires_at,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            conn.commit()

    def get_user_by_token(
        self,
        token: str,
    ) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    users.id,
                    users.username,
                    auth_tokens.expires_at
                FROM auth_tokens
                INNER JOIN users
                    ON users.id = auth_tokens.user_id
                WHERE auth_tokens.token = ?
                """,
                (token,),
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    def delete_token(self, token: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                DELETE FROM auth_tokens
                WHERE token = ?
                """,
                (token,),
            )
            conn.commit()

class AuthRepository:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
           self.path,
           timeout=60.0,
           check_same_thread=False,
     )

        connection.execute("PRAGMA busy_timeout = 60000")

        connection.row_factory = sqlite3.Row
        return connection

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    recovery_code_hash TEXT,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_tokens (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id)
                        REFERENCES users(id)
                        ON DELETE CASCADE
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_auth_tokens_user_id
                ON auth_tokens(user_id)
                """
            )

            try:
                conn.execute(
                    "ALTER TABLE users "
                    "ADD COLUMN recovery_code_hash TEXT"
                )
            except sqlite3.OperationalError:
                pass

            conn.commit()

    def create_user(
        self,
        username: str,
        password_hash: str,
        password_salt: str,
        role: str = "user",
        recovery_code_hash: str | None = None,
    ) -> dict:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users(
                    username,
                    password_hash,
                    password_salt,
                    recovery_code_hash,
                    role,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    username,
                    password_hash,
                    password_salt,
                    recovery_code_hash,
                    role,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            conn.commit()

            user_id = cursor.lastrowid

        return {
            "id": user_id,
            "username": username,
            "role": role,
        }

    def get_user_by_username(
        self,
        username: str,
    ) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    id,
                    username,
                    password_hash,
                    password_salt,
                    recovery_code_hash,
                    role,
                    created_at
                FROM users
                WHERE username = ?
                """,
                (username,),
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    def update_password(
        self,
        user_id: int,
        password_hash: str,
        password_salt: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                '''
                UPDATE users
                SET password_hash = ?,
                    password_salt = ?
                WHERE id = ?
                ''',
                (
                    password_hash,
                    password_salt,
                    user_id,
                ),
            )

            conn.execute(
                '''
                DELETE FROM auth_tokens
                WHERE user_id = ?
                ''',
                (user_id,),
            )

            conn.commit()

    def save_token(
        self,
        token: str,
        user_id: int,
        expires_at: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO auth_tokens(
                    token,
                    user_id,
                    expires_at,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    token,
                    user_id,
                    expires_at,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            conn.commit()

    def get_user_by_token(
        self,
        token: str,
    ) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    users.id,
                    users.username,
                    users.role,
                    auth_tokens.expires_at
                FROM auth_tokens
                INNER JOIN users
                    ON users.id = auth_tokens.user_id
                WHERE auth_tokens.token = ?
                """,
                (token,),
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    def delete_token(
        self,
        token: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                DELETE FROM auth_tokens
                WHERE token = ?
                """,
                (token,),
            )

            conn.commit()