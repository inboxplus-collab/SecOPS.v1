from __future__ import annotations

from datetime import datetime
from typing import List

from .base import BaseCheck, CheckContext, CheckResult
from integrations.ci.github_actions import GitHubActionsClient


class CIHardeningCheck(BaseCheck):
    id = "ci_hardening"
    name = "CI pipeline hardening"
    description = "Checks for dangerous patterns in GitHub Actions workflows."

    async def run(self, ctx: CheckContext) -> List[CheckResult]:
        client = GitHubActionsClient.from_env()
        # In real-world, you'd get org/repo from ctx, here we assume they are passed in metadata.
        repo_full_name = ctx.extra.get("github_repo") if ctx.extra else None
        if not repo_full_name:
            return []

        workflows = await client.list_workflows(repo_full_name)
        results: List[CheckResult] = []

        for wf in workflows:
            found_dangerous = False
            reasons: list[str] = []

            if "pull_request_target" in wf.raw_yaml:
                found_dangerous = True
                reasons.append("Uses 'pull_request_target' which is risky for untrusted forks.")

            if "secrets.GITHUB_TOKEN" in wf.raw_yaml and "permissions:" not in wf.raw_yaml:
                found_dangerous = True
                reasons.append("GITHUB_TOKEN used without explicit permission scoping.")

            if found_dangerous:
                results.append(
                    CheckResult(
                        check_id=self.id,
                        title=f"Potentially unsafe CI workflow: {wf.name}",
                        description="\n".join(reasons),
                        severity="high",
                        metadata={
                            "workflow_id": wf.id,
                            "workflow_name": wf.name,
                            "path": wf.path,
                        },
                        detected_at=datetime.utcnow(),
                    )
                )

        return results
