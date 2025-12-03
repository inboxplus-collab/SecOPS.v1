from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.audit.models import AuditLog


class AuditLogger:
    @staticmethod
    async def log(
        session: AsyncSession, *, user_id: str, action: str, metadata: Optional[Dict[str, Any]] = None
    ) -> AuditLog:
        """Persist an immutable audit log entry for compliance visibility."""

        entry = AuditLog(
            user_id=user_id,
            action=action,
            metadata=metadata or {},
            timestamp=datetime.utcnow(),
        )
        session.add(entry)
        await session.commit()
        await session.refresh(entry)
        return entry

    @staticmethod
    async def query(session: AsyncSession, limit: int = 200) -> List[AuditLog]:
        """Return the most recent audit log entries, newest first."""

        stmt = select(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())
