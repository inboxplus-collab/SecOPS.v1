from __future__ import annotations

from fastapi import HTTPException, status


class SecOpsError(Exception):
    """Base app-level exception."""


class NotFoundError(SecOpsError):
    """Raised when a resource cannot be found."""


class PermissionDeniedError(SecOpsError):
    """Raised when a user is not allowed to perform an action."""


class ExternalServiceError(SecOpsError):
    """Raised when an external system (GitHub, K8s, scanner) fails."""


def http_404(detail: str = "Not found") -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


def http_403(detail: str = "Forbidden") -> HTTPException:
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


def http_400(detail: str = "Bad request") -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
