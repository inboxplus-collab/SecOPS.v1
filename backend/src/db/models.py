# backend/src/db/models.py
"""
Central SQLAlchemy ORM base and model registration.

This module defines the global Declarative Base class that all ORM models
inherit from, and serves as the single place where Alembic and the rest of
the application discover metadata for migrations.

Usage:
    from db.models import Base
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase, declared_attr

# ---------------------------------------------------------------------------
# Naming convention for constraints / indexes
# ---------------------------------------------------------------------------

NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix__%(table_name)s__%(column_0_N_name)s",
    "uq": "uq__%(table_name)s__%(column_0_N_name)s",
    "ck": "ck__%(table_name)s__%(constraint_name)s",
    "fk": "fk__%(table_name)s__%(column_0_N_name)s__%(referred_table_name)s",
    "pk": "pk__%(table_name)s",
}

metadata_obj = MetaData(naming_convention=NAMING_CONVENTION)


class Base(DeclarativeBase):
    """
    Global SQLAlchemy Declarative Base for all ORM models.

    All ORM model classes (e.g. Issue, User, Org) must inherit from this Base
    so that:
      - Alembic can see them via Base.metadata
      - db.session / engine helpers can create/drop tables for local dev
    """

    metadata = metadata_obj

    @declared_attr.directive
    def __tablename__(cls) -> str:  # type: ignore[override]
        # CamelCase → snake_case by default if not explicitly set.
        name = cls.__name__
        snake: list[str] = []
        for i, ch in enumerate(name):
            if ch.isupper() and i != 0:
                snake.append("_")
            snake.append(ch.lower())
        return "".join(snake)


# ---------------------------------------------------------------------------
# Model registration
# ---------------------------------------------------------------------------

try:
    from core.issues.models import Issue  # noqa: F401
    from core.audit.models import AuditLog  # noqa: F401
except Exception:
    # In early setup or some tests the module may not be importable yet.
    Issue = AuditLog = Any  # type: ignore[assignment]

__all__ = ["Base", "metadata_obj"]
