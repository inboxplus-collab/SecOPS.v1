# backend/src/api/deps.py

from __future__ import annotations

import logging
from typing import Any, Dict, Generator

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from sqlalchemy.orm import Session

from integrations.supabase_auth import decode_supabase_jwt, SupabaseAuthError
from db.session import get_db  # this is your migrations/session.py helper
from rag.SearchOrchestrator import SearchOrchestrator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AUTH HELPERS
# ---------------------------------------------------------------------------

bearer_scheme = HTTPBearer(auto_error=False)

# Allow routes to opt into softer authentication when validation should run first
optional_bearer = HTTPBearer(auto_error=False)


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


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> Dict[str, Any]:
    """Alias wrapper for compatibility with tests and routers."""

    return await get_supabase_user(credentials)


async def get_optional_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(optional_bearer),
) -> Dict[str, Any] | None:
    """Return a user dict when credentials are present, otherwise None."""

    override = request.app.dependency_overrides.get(get_current_user)
    if override:
        return override()

    if credentials is None:
        return None

    return await get_supabase_user(credentials)


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


def get_rag_orchestrator() -> Any:
    """Provide a lightweight RAG orchestrator usable in dependency overrides."""

    class _Wrapper:
        def __init__(self) -> None:
            self.engine = SearchOrchestrator()

        async def run(
            self,
            *,
            org_id: str,
            query: str,
            max_tokens: int | None = None,
            temperature: float | None = None,
            top_k: int | None = None,
            metadata: Dict[str, Any] | None = None,
        ) -> Dict[str, Any]:
            _ = (max_tokens, temperature, top_k)  # unused but kept for API compatibility
            answer = await self.engine.query(query, metadata=metadata or {})
            return {
                "answer": answer.answer if hasattr(answer, "answer") else str(answer),
                "sources": getattr(answer, "citations", []),
                "usage": getattr(answer, "usage", None),
            }

    return _Wrapper()


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
    "get_current_user",
    "get_optional_current_user",
    "get_rag_orchestrator",
    "require_admin",
    "db_session",
    "get_request_context",
]
