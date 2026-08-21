from __future__ import annotations

from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row


class PostgresAuthRepository:
    def __init__(
        self,
        host: str,
        database: str,
        user: str,
        password: str,
    ) -> None:
        if not all(
            [
                host.strip(),
                database.strip(),
                user.strip(),
                password,
            ]
        ):
            raise ValueError(
                "PostgreSQL authentication settings are incomplete."
            )

        self.host = host.strip()
        self.database = database.strip()
        self.user = user.strip()
        self.password = password
        self._initialized = False

    def connect(self):
        return psycopg.connect(
            host=self.host,
            dbname=self.database,
            user=self.user,
            password=self.password,
            port=5432,
            sslmode="require",
            connect_timeout=15,
            row_factory=dict_row,
        )

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        self.init_db()
        self._initialized = True

    def init_db(self) -> None:
        with self.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id BIGSERIAL PRIMARY KEY,
                        username TEXT NOT NULL UNIQUE,
                        password_hash TEXT NOT NULL,
                        password_salt TEXT NOT NULL,
                        recovery_code_hash TEXT,
                        role TEXT NOT NULL DEFAULT 'user',
                        created_at TEXT NOT NULL
                    )
                    """
                )

                cursor.execute(
                    """
                    ALTER TABLE users
                    ADD COLUMN IF NOT EXISTS
                    recovery_code_hash TEXT
                    """
                )

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS auth_tokens (
                        token TEXT PRIMARY KEY,
                        user_id BIGINT NOT NULL
                            REFERENCES users(id)
                            ON DELETE CASCADE,
                        expires_at TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_auth_tokens_user_id
                    ON auth_tokens(user_id)
                    """
                )

            conn.commit()

    def create_user(
        self,
        username: str,
        password_hash: str,
        password_salt: str,
        role: str = "user",
        recovery_code_hash: str | None = None,
    ) -> dict:
        self._ensure_initialized()

        with self.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO users (
                        username,
                        password_hash,
                        password_salt,
                        recovery_code_hash,
                        role,
                        created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        username,
                        password_hash,
                        password_salt,
                        recovery_code_hash,
                        role,
                        datetime.now(
                            timezone.utc
                        ).isoformat(),
                    ),
                )

                row = cursor.fetchone()

            conn.commit()

        if row is None:
            raise RuntimeError(
                "PostgreSQL did not return a user ID."
            )

        return {
            "id": row["id"],
            "username": username,
            "role": role,
        }

    def get_user_by_username(
        self,
        username: str,
    ) -> dict | None:
        self._ensure_initialized()

        with self.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
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
                    WHERE username = %s
                    """,
                    (username,),
                )

                row = cursor.fetchone()

        return row

    def update_password(
        self,
        user_id: int,
        password_hash: str,
        password_salt: str,
    ) -> None:
        self._ensure_initialized()

        with self.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE users
                    SET password_hash = %s,
                        password_salt = %s
                    WHERE id = %s
                    """,
                    (
                        password_hash,
                        password_salt,
                        user_id,
                    ),
                )

                cursor.execute(
                    """
                    DELETE FROM auth_tokens
                    WHERE user_id = %s
                    """,
                    (user_id,),
                )

            conn.commit()

    def save_token(
        self,
        token: str,
        user_id: int,
        expires_at: str,
    ) -> None:
        self._ensure_initialized()

        with self.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO auth_tokens (
                        token,
                        user_id,
                        expires_at,
                        created_at
                    )
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        token,
                        user_id,
                        expires_at,
                        datetime.now(
                            timezone.utc
                        ).isoformat(),
                    ),
                )

            conn.commit()

    def get_user_by_token(
        self,
        token: str,
    ) -> dict | None:
        self._ensure_initialized()

        with self.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        users.id,
                        users.username,
                        users.role,
                        auth_tokens.expires_at
                    FROM auth_tokens
                    INNER JOIN users
                        ON users.id = auth_tokens.user_id
                    WHERE auth_tokens.token = %s
                    """,
                    (token,),
                )

                row = cursor.fetchone()

        return row

    def delete_token(
        self,
        token: str,
    ) -> None:
        self._ensure_initialized()

        with self.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM auth_tokens
                    WHERE token = %s
                    """,
                    (token,),
                )

            conn.commit()