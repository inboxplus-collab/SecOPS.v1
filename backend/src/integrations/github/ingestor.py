from __future__ import annotations

from typing import List

from sqlalchemy.ext.asyncio import AsyncSession

from core.checks.base import CheckContext
from core.checks.ci_hardening import CIHardeningCheck
from core.issues.service import create_issue
from .mappers import check_result_to_issue_payload


async def run_github_checks_for_repo(db: AsyncSession, repo_full_name: str) -> List[str]:
    """
    Run GitHub-specific checks (CI hardening) and create issues.

    Returns list of issue IDs created.
    """
    ctx = CheckContext(extra={"github_repo": repo_full_name})
    check = CIHardeningCheck()
    results = await check.run(ctx)
    ids: List[str] = []
    for r in results:
        payload_dict = check_result_to_issue_payload(r)
        issue = await create_issue(db, payload_dict)  # type: ignore[arg-type]
        ids.append(issue.id)
    return ids
