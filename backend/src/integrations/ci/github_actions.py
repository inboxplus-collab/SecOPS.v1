# backend/src/integrations/ci/github_actions.py

"""
GitHub Actions CI integration.

Capabilities:
- List workflow runs
- Find latest failed workflow run
- Fetch high-level job/step logs for a run

Relies on:
- GITHUB_TOKEN (PAT or fine-grained token)
- TARGET_REPO (e.g. "inboxplus-collab/SecOPS.v1")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
import os


GITHUB_API = "https://api.github.com"


class GitHubActionsClient:
    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")
        self.repo = os.getenv("TARGET_REPO")

        if not self.repo:
            raise RuntimeError("TARGET_REPO env var is required (e.g. 'owner/repo').")

        self.headers = {
            "Accept": "application/vnd.github+json",
        }
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"

    async def list_workflow_runs(
        self,
        branch: Optional[str] = None,
        status: Optional[str] = None,
        per_page: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        List workflow runs for the repo.

        status can be: "success", "failure", "cancelled", etc.
        """
        params: Dict[str, Any] = {"per_page": per_page}
        if branch:
            params["branch"] = branch
        if status:
            params["status"] = status

        url = f"{GITHUB_API}/repos/{self.repo}/actions/runs"

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=self.headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        return data.get("workflow_runs", [])

    async def get_latest_failed_run(
        self, branch: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        runs = await self.list_workflow_runs(branch=branch, status="failure", per_page=5)
        if not runs:
            return None
        # Already sorted newest-first by GitHub
        return runs[0]

    async def get_run_jobs(self, run_id: int) -> List[Dict[str, Any]]:
        """
        Fetch jobs and steps for a workflow run.
        This is much lighter than downloading raw logs tarball.
        """
        url = f"{GITHUB_API}/repos/{self.repo}/actions/runs/{run_id}/jobs"

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(url, headers=self.headers)
            resp.raise_for_status()
            data = resp.json()

        return data.get("jobs", [])

    async def summarize_run_logs(self, run_id: int) -> str:
        """
        Build a textual summary of jobs/steps for feeding into the LLM.
        """
        jobs = await self.get_run_jobs(run_id)
        lines: List[str] = []

        for job in jobs:
            lines.append(
                f"JOB: {job.get('name')} (id={job.get('id')}) "
                f"status={job.get('status')} conclusion={job.get('conclusion')}"
            )
            for step in job.get("steps", []):
                lines.append(
                    f"  STEP: {step.get('name')} "
                    f"status={step.get('status')} conclusion={step.get('conclusion')}"
                )

        return "\n".join(lines)


github_actions_client = GitHubActionsClient()
