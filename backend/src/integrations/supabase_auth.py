"""Supabase JWT decoding helpers used by FastAPI dependencies."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer
from jose import JWTError, jwt

from utils.config import settings


class SupabaseAuthError(Exception):
    """Raised when Supabase JWT validation fails."""


def decode_supabase_jwt(token: str) -> Dict[str, Any]:
    """Decode and validate a Supabase JWT using the configured secret."""

    if not token:
        raise SupabaseAuthError("Missing token")

    try:
        return jwt.decode(
            token,
            settings.SUPABASE_JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
            options={"verify_aud": False},
        )
    except JWTError as exc:
        raise SupabaseAuthError(str(exc)) from exc


auth_scheme = HTTPBearer()


async def verify_supabase_jwt(request: Request) -> Dict[str, Any]:
    """Extract a token from the request and attach the decoded user payload."""

    token = None

    # 1. Authorization header
    if "authorization" in request.headers:
        parts = request.headers["authorization"].split(" ")
        if len(parts) == 2:
            token = parts[1]

    # 2. Cookie fallback
    if not token:
        token = request.cookies.get("sb-access-token")

    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authentication token")

    try:
        payload = decode_supabase_jwt(token)
        request.state.user = payload
        return payload
    except SupabaseAuthError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Supabase JWT")


__all__ = ["decode_supabase_jwt", "verify_supabase_jwt", "SupabaseAuthError", "auth_scheme"]
