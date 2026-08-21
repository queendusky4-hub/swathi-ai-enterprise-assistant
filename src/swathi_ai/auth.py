from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field

from .database import AuthRepository


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/login",
)


class RegisterRequest(BaseModel):
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
    )

    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
    )


class ResetPasswordRequest(BaseModel):
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
    )

    recovery_code: str = Field(
        ...,
        min_length=8,
        max_length=128,
    )

    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
    )


class RegisterResponse(BaseModel):
    id: int
    username: str
    role: str
    recovery_code: str


class UserResponse(BaseModel):
    id: int
    username: str
    role: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AuthService:
    def __init__(
        self,
        repository: AuthRepository,
        token_expiry_hours: int = 24,
    ) -> None:
        self.repository = repository
        self.token_expiry_hours = token_expiry_hours
        self._guest_tokens: dict[str, datetime] = {}

    @staticmethod
    def hash_password(
        password: str,
        salt: bytes,
    ) -> bytes:
        return hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            310_000,
        )

    def register(
        self,
        username: str,
        password: str,
    ) -> dict:
        clean_username = username.strip().lower()

        if not re.fullmatch(
            r"[a-zA-Z0-9_.-]{3,50}",
            clean_username,
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Username may contain only letters, numbers, "
                    "underscores, dots and hyphens."
                ),
            )

        if self.repository.get_user_by_username(
            clean_username
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already exists.",
            )

        salt = secrets.token_bytes(32)

        password_hash = self.hash_password(
            password,
            salt,
        )

        recovery_code = secrets.token_urlsafe(12)

        recovery_hash = hashlib.sha256(
            recovery_code.encode("utf-8")
        ).hexdigest()

        user = self.repository.create_user(
            username=clean_username,
            password_hash=password_hash.hex(),
            password_salt=salt.hex(),
            role="user",
            recovery_code_hash=recovery_hash,
        )

        user["recovery_code"] = recovery_code

        return user

    def reset_password(
        self,
        username: str,
        recovery_code: str,
        new_password: str,
    ) -> None:
        clean_username = username.strip().lower()

        user = self.repository.get_user_by_username(
            clean_username
        )

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid username or recovery code.",
            )

        stored_recovery_hash = user.get(
            "recovery_code_hash"
        )

        supplied_recovery_hash = hashlib.sha256(
            recovery_code.strip().encode("utf-8")
        ).hexdigest()

        if (
            not stored_recovery_hash
            or not hmac.compare_digest(
                stored_recovery_hash,
                supplied_recovery_hash,
            )
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid username or recovery code.",
            )

        new_salt = secrets.token_bytes(32)

        new_password_hash = self.hash_password(
            new_password,
            new_salt,
        )

        self.repository.update_password(
            user_id=user["id"],
            password_hash=new_password_hash.hex(),
            password_salt=new_salt.hex(),
        )

    def authenticate(
        self,
        username: str,
        password: str,
    ) -> dict:
        clean_username = username.strip().lower()

        user = self.repository.get_user_by_username(
            clean_username
        )

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password.",
                headers={
                    "WWW-Authenticate": "Bearer",
                },
            )

        salt = bytes.fromhex(user["password_salt"])

        expected_hash = bytes.fromhex(
            user["password_hash"]
        )

        supplied_hash = self.hash_password(
            password,
            salt,
        )

        if not hmac.compare_digest(
            expected_hash,
            supplied_hash,
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password.",
                headers={
                    "WWW-Authenticate": "Bearer",
                },
            )

        return user

    def create_access_token(
        self,
        user_id: int,
    ) -> str:
        token = secrets.token_urlsafe(48)

        expires_at = (
            datetime.now(timezone.utc)
            + timedelta(hours=self.token_expiry_hours)
        )

        self.repository.save_token(
            token=token,
            user_id=user_id,
            expires_at=expires_at.isoformat(),
        )

        return token

    def create_guest_access_token(self) -> str:
        token = "guest_" + secrets.token_urlsafe(48)

        expires_at = (
            datetime.now(timezone.utc)
            + timedelta(hours=self.token_expiry_hours)
        )

        self._guest_tokens[token] = expires_at

        return token

    def get_user_from_token(
        self,
        token: str,
    ) -> dict | None:
        guest_expiry = self._guest_tokens.get(token)

        if guest_expiry is not None:
            if guest_expiry <= datetime.now(timezone.utc):
                self._guest_tokens.pop(token, None)
                return None

            return {
                "id": 0,
                "username": "Guest",
                "role": "guest",
                "expires_at": guest_expiry.isoformat(),
            }

        user = self.repository.get_user_by_token(token)

        if user is None:
            return None

        expires_at = datetime.fromisoformat(
            user["expires_at"]
        )

        if expires_at <= datetime.now(timezone.utc):
            self.repository.delete_token(token)
            return None

        return user


def create_current_user_dependency(
    auth_service: AuthService,
):
    def get_current_user(
        token: str = Depends(oauth2_scheme),
    ) -> dict:
        user = auth_service.get_user_from_token(token)

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired login session.",
                headers={
                    "WWW-Authenticate": "Bearer",
                },
            )

        return user

    return get_current_user