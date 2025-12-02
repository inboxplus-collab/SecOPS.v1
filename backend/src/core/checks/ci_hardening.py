from __future__ import annotations

from datetime import datetime
from typing import Any, List

from api.schemas.issues import IssueSeverity, IssueStatus, IssueLocation
from .base import BaseCheck, CheckContext, CheckRunResult, LoggerLike
from integrations.ci.github_actions import GitHubActionsClient


class CIHardeningCheck(BaseCheck):
    id = "ci_hardening"
    name = "CI pipeline hardening"
    description = "Checks for dangerous patterns in GitHub Actions workflows."

    def __init__(self) -> None:
        super().__init__(default_severity=IssueSeverity.high)

    async def run(self, context: CheckContext, logger: LoggerLike) -> CheckRunResult:
        client = GitHubActionsClient.from_env()
        repo_full_name = context.get_extra("github_repo")
        if not repo_full_name:
            return CheckRunResult(
                issues=[],
                metrics={"skipped": True, "reason": "missing_repo"},
                errors=[],
            )

        workflows = await client.list_workflows(repo_full_name)
        issues: List[Any] = []
        errors: List[str] = []

        for wf in workflows:
            reasons: list[str] = []

            if "pull_request_target" in wf.raw_yaml:
                reasons.append("Uses 'pull_request_target' which is risky for untrusted forks.")

            if "secrets.GITHUB_TOKEN" in wf.raw_yaml and "permissions:" not in wf.raw_yaml:
                reasons.append("GITHUB_TOKEN used without explicit permission scoping.")

            if reasons:
                issue_id = f"{context.org_id}:{wf.id}"
                location = IssueLocation(kind="ci", file_path=wf.path, repo=repo_full_name)
                issue = self.build_issue(
                    id=issue_id,
                    org_id=context.org_id or "unknown-org",
                    title=f"Potentially unsafe CI workflow: {wf.name}",
                    severity=self.default_severity,
                    status=IssueStatus.open,
                    service=repo_full_name,
                    category="ci",
                    tags=["ci", "github", "workflow", "security"],
                    location=location,
                    source="github",
                    short_description="\n".join(reasons),
                    description="\n".join(reasons),
                    root_cause="Workflow configuration may allow privilege escalation.",
                    impact="Malicious pull requests could access secrets or modify pipelines.",
                    proposed_fix="Avoid pull_request_target for untrusted code and scope GITHUB_TOKEN permissions.",
                    precautions="Review workflow triggers and token permissions before merging.",
                    references=[],
                    extra={"workflow_name": wf.name},
                )
                issues.append(issue)

        metrics = {
            "workflows_scanned": len(workflows),
            "issues_found": len(issues),
            "repo": repo_full_name,
        }

        return CheckRunResult(issues=issues, metrics=metrics, errors=errors)
