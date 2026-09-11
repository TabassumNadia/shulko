"""
Authentication primitives — password hashing and session tokens.

Kept deliberately small: bcrypt for hashing (never store or log a plain
password), PyJWT for a stateless session token (no server-side session
table to keep in sync). Everything that needs the currently-logged-in
user reads it from this token via `backend.main.get_current_user`.
"""

from __future__ import annotations

import datetime as dt

import bcrypt
import jwt

from backend.core.config import get_settings

ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    """Bcrypt handles its own salt; the hash alone is enough to verify."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # A malformed hash must fail closed, not raise past the caller.
        return False


def create_access_token(user_id: int, email: str) -> str:
    settings = get_settings()
    expire = dt.datetime.now(dt.timezone.utc) + dt.timedelta(
        minutes=settings.jwt_expire_minutes
    )
    payload = {"sub": str(user_id), "email": email, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    """Returns the payload, or None for anything expired/tampered/malformed.

    Never raises: every caller treats None as "not authenticated" rather
    than juggling a jwt-library-specific exception.
    """
    settings = get_settings()
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None
