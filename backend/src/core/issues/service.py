from __future__ import annotations

from typing import Any, Dict, Optional

from api.schemas.issues import IssueDetail, IssueResolutionState
from core.issues.repository import IssuesRepository
from db.session import get_session_maker


class IssuesService:
    """
    Application-facing service for querying and mutating issues.

    This wraps the repository layer with session management and transforms
    ORM models into response DTOs used by the API layer.
    """

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.repo = IssuesRepository()
        self._session_maker = get_session_maker()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def list_issues(
        self,
        *,
        org_id: str,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        service: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        async with self._session_maker() as session:
            items = await self.repo.list_issues(
                session,
                org_id=org_id,
                status=status,
                severity=severity,
                service=service,
                limit=limit,
                offset=offset,
            )
            total = await self.repo.count_issues(
                session,
                org_id=org_id,
                status=status,
                severity=severity,
                service=service,
            )

        summaries = [issue.to_summary() for issue in items]
        return {
            "items": summaries,
            "total": total,
            "page": (offset // limit) + 1,
            "page_size": limit,
        }

    async def get_issue(self, *, org_id: str, issue_id: str) -> Optional[Dict[str, Any]]:
        async with self._session_maker() as session:
            issue = await self.repo.get_issue(session, org_id=org_id, issue_id=issue_id)
            if issue is None:
                return None
            detail: IssueDetail = issue.to_detail()
            return detail.model_dump()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    async def resolve_issue(
        self,
        *,
        org_id: str,
        issue_id: str,
        resolution_state: IssueResolutionState,
        resolution_note: Optional[str],
        resolved_by: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        async with self._session_maker() as session:
            issue = await self.repo.update_issue_resolution(
                session,
                org_id=org_id,
                issue_id=issue_id,
                resolution_state=resolution_state,
                resolution_note=resolution_note,
                resolved_by=resolved_by,
            )
            if issue is None:
                return None
            await session.commit()
            return issue.to_detail().model_dump()

    async def upsert_issues_from_check_run(self, *, issues_data: list[dict[str, Any]]) -> Dict[str, int]:
        async with self._session_maker() as session:
            created, updated = await self.repo.upsert_many(session, issues_data)
            await session.commit()
            return {"created": created, "updated": updated}

