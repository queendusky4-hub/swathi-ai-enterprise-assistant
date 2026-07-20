from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, Field

#oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")
# =========================================================
# Authentication configuration
# =========================================================

SECRET_KEY = "change-this-secret-key-before-deployment"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


password_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/login",
)


# =========================================================
# Schemas
# =========================================================


class RegisterRequest(BaseModel):
    name: str = Field(
        ...,
        min_length=2,
        max_length=100,
    )

    email: EmailStr

    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
    )


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserRecord(BaseModel):
    id: int
    name: str
    email: EmailStr
    password_hash: str
    role: str = "user"
    is_active: bool = True
    created_at: datetime


class UserResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
    role: str
    is_active: bool
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenPayload(BaseModel):
    sub: str
    user_id: int
    role: str
    exp: int


# =========================================================
# Temporary user storage
# =========================================================

users: dict[str, UserRecord] = {}


# =========================================================
# Password utilities
# =========================================================


def hash_password(password: str) -> str:
    return password_context.hash(password)


def verify_password(
    plain_password: str,
    password_hash: str,
) -> bool:
    try:
        return password_context.verify(
            plain_password,
            password_hash,
        )
    except Exception:
        return False


# =========================================================
# Token utilities
# =========================================================


def create_access_token(
    user: UserRecord,
) -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES,
    )

    payload = {
        "sub": user.email,
        "user_id": user.id,
        "role": user.role,
        "exp": expires_at,
    }

    return jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


def decode_access_token(
    token: str,
) -> TokenPayload:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired access token.",
        headers={
            "WWW-Authenticate": "Bearer",
        },
    )

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
        )

        subject = payload.get("sub")
        user_id = payload.get("user_id")
        role = payload.get("role")
        expiration = payload.get("exp")

        if not isinstance(subject, str):
            raise credentials_error

        if not isinstance(user_id, int):
            raise credentials_error

        if not isinstance(role, str):
            raise credentials_error

        if not isinstance(expiration, int):
            raise credentials_error

        return TokenPayload(
            sub=subject,
            user_id=user_id,
            role=role,
            exp=expiration,
        )

    except JWTError as error:
        raise credentials_error from error


# =========================================================
# User utilities
# =========================================================


def get_user_by_email(
    email: str,
) -> UserRecord | None:
    normalized_email = email.lower().strip()
    return users.get(normalized_email)

def register_user(
    request: RegisterRequest,
) -> UserRecord:
    normalized_email = request.email.lower().strip()

    if normalized_email in users:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists.",
        )

    user = UserRecord(
        id=len(users) + 1,
        name=request.name.strip(),
        email=normalized_email,
        password_hash=hash_password(request.password),
        role="user",
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )

    users[normalized_email] = user

    return user


def authenticate_user(
    email: str,
    password: str,
) -> UserRecord:
    normalized_email = email.lower().strip()

    user = get_user_by_email(normalized_email)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    if not verify_password(
        password,
        user.password_hash,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )

    return user


# =========================================================
# FastAPI dependencies
# =========================================================


def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> UserRecord:
    token_payload = decode_access_token(token)

    user = get_user_by_email(token_payload.sub)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer exists.",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )

    return user


def require_admin(
    user: Annotated[
        UserRecord,
        Depends(get_current_user),
    ],
) -> UserRecord:
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access is required.",
        )

    return user


# =========================================================
# Response helpers
# =========================================================


def build_user_response(
    user: UserRecord,
) -> UserResponse:
    return UserResponse(
        id=user.id,
        name=user.name,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
    )


def build_token_response(
    user: UserRecord,
) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(user),
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )