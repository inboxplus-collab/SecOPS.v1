# backend/src/db/session.py

import logging

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from utils.config import settings

logger = logging.getLogger(__name__)


RAW_DATABASE_URL = settings.DATABASE_URL

if RAW_DATABASE_URL.startswith("sqlite"):
    DATABASE_URL = RAW_DATABASE_URL.replace("sqlite://", "sqlite+aiosqlite://")
    engine_kwargs = {
        "echo": settings.DEBUG,
        "pool_pre_ping": False,
        "connect_args": {"check_same_thread": False},
    }
else:
    DATABASE_URL = RAW_DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    engine_kwargs = {
        "echo": settings.DEBUG,
        "pool_size": 10,
        "max_overflow": 20,
        "pool_pre_ping": True,
    }


try:
    engine = create_async_engine(
        DATABASE_URL,
        **engine_kwargs,
    )

    SessionLocal = async_sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        expire_on_commit=False,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - environment safeguard
    logger.warning("Async database driver unavailable; DB helpers disabled: %s", exc)
    engine = None
    SessionLocal = None


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    if SessionLocal is None:
        raise RuntimeError("Database engine is not initialized; ensure driver dependencies are installed.")

    async with SessionLocal() as session:
        yield session
