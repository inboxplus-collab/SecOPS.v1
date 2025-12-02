# backend/src/api/deps.py

from __future__ import annotations

import logging
from typing import Any, Dict, Generator

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from sqlalchemy.orm import Session

from integrations.supabase_auth import decode_supabase_jwt, SupabaseAuthError
from db.session import get_db  # this is your migrations/session.py helper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AUTH HELPERS
# ---------------------------------------------------------------------------

bearer_scheme = HTTPBearer(auto_error=False)


async def get_supabase_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> Dict[str, Any]:
    """
    Validate a Supabase JWT access token and return a normalized user payload.

    Accepts:
        Authorization: Bearer <token>

    Returns:
        {
          "id": user_id,
          "email": email,
          "role": "authenticated" | "service_role" | "anon",
          "raw": <full jwt payload>
        }
    """

    # Missing header
    if credentials is None or not credentials.scheme or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header",
        )

    token = credentials.credentials

    # Validate token through Supabase decoder
    try:
        payload = decode_supabase_jwt(token)
    except SupabaseAuthError as exc:
        logger.warning("Supabase JWT decode failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired Supabase token",
        )

    # Required
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Supabase token missing subject (sub)",
        )

    # Fields
    return {
        "id": user_id,
        "email": payload.get("email"),
        "role": payload.get("role", "authenticated"),
        "raw": payload,
    }


# ---------------------------------------------------------------------------
# ADMIN GUARD
# ---------------------------------------------------------------------------

async def require_admin(user: Dict[str, Any] = Depends(get_supabase_user)) -> Dict[str, Any]:
    """
    Protect endpoints for admin-only access.
    Supabase default roles: authenticated, service_role, anon
    """
    role = user.get("role")

    if role not in ("service_role", "admin", "superadmin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )

    return user


# ---------------------------------------------------------------------------
# DATABASE DEPENDENCY
# ---------------------------------------------------------------------------

def db_session() -> Generator[Session, None, None]:
    """
    Provide a SQLAlchemy database session to the route.

    Usage:
        def route(db: Session = Depends(db_session)):
            return db.execute(select(MyModel)).scalars().all()
    """
    try:
        db = get_db()
        session = next(db)
        yield session
        session.commit()
    except Exception:  # noqa: BLE001
        logger.exception("DB session error – rolling back")
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# OPTIONAL: GENERIC REQUEST CONTEXT
# ---------------------------------------------------------------------------

async def get_request_context(
    request: Request,
    user: Dict[str, Any] = Depends(get_supabase_user),
) -> Dict[str, Any]:
    """
    Attach user + headers + ip to context, useful for logs & RAG context.
    """
    return {
        "user": user,
        "headers": dict(request.headers),
        "client_ip": request.client.host if request.client else None,
        "path": request.url.path,
        "method": request.method,
    }


__all__ = [
    "get_supabase_user",
    "require_admin",
    "db_session",
    "get_request_context",
]
