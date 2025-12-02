from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import sessionmaker

from db.models import Base
from utils.config import settings

_engine: AsyncEngine | None = None
_session_maker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url,
            echo=settings.env == "development",
            future=True,
        )
    return _engine


def get_session_maker() -> async_sessionmaker[AsyncSession]:
    global _session_maker
    if _session_maker is None:
        _session_maker = async_sessionmaker(
            bind=get_engine(),
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_maker


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency:

        async def route(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(MyModel))
            return result.scalars().all()

    """
    session = get_session_maker()()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def create_all() -> None:
    """
    Utility to create all DB tables (for dev/testing).
    """
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
