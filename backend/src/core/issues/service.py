from __future__ import annotations

from typing import List, Optional, Tuple

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from core.issues.models import Issue
from api.schemas.issues import IssueCreate, IssueUpdate
from utils.errors import NotFoundError


async def create_issue(db: AsyncSession, payload: IssueCreate) -> Issue:
    issue = Issue(
        title=payload.title,
        description=payload.description,
        severity=payload.severity,
        status=payload.status,
        source=payload.source,
        metadata=payload.metadata,
    )
    db.add(issue)
    await db.flush()
    await db.refresh(issue)
    return issue


async def list_issues(
    db: AsyncSession,
    page: int = 1,
    page_size: int = 20,
    status: Optional[str] = None,
    severity: Optional[str] = None,
) -> Tuple[List[Issue], int]:
    query = select(Issue)
    count_query = select(func.count(Issue.id))

    if status:
        query = query.where(Issue.status == status)
        count_query = count_query.where(Issue.status == status)
    if severity:
        query = query.where(Issue.severity == severity)
        count_query = count_query.where(Issue.severity == severity)

    total = (await db.execute(count_query)).scalar_one()
    query = query.order_by(Issue.detected_at.desc()).offset((page - 1) * page_size).limit(page_size)
    rows = (await db.execute(query)).scalars().all()
    return rows, total


async def get_issue(db: AsyncSession, issue_id: str) -> Issue:
    issue = await db.get(Issue, issue_id)
    if not issue:
        raise NotFoundError(f"Issue {issue_id} not found")
    return issue


async def update_issue(db: AsyncSession, issue_id: str, payload: IssueUpdate) -> Issue:
    issue = await get_issue(db, issue_id)
    data = payload.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(issue, k, v)
    await db.flush()
    await db.refresh(issue)
    return issue
