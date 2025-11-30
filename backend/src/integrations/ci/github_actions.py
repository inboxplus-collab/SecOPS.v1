from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import httpx


@dataclass
class WorkflowFile:
    id: int
    name: str
    path: str
    raw_yaml: str


class GitHubActionsClient:
    """
    Minimal GitHub Actions client used for CI hardening checks.
    """

    def __init__(self, token: str, base_url: str = "https://api.github.com") -> None:
        self._token = token
        self._base_url = base_url
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/vnd.github+json",
            },
            timeout=20.0,
        )

    @classmethod
    def from_env(cls) -> "GitHubActionsClient":
        token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or ""
        if not token:
            raise RuntimeError("GITHUB_TOKEN not configured in environment")
        return cls(token=token)

    async def list_workflows(self, repo_full_name: str) -> List[WorkflowFile]:
        """
        List workflow YAMLs for a repository.

        repo_full_name: "owner/repo"
        """
        owner, repo = repo_full_name.split("/", 1)
        # First get workflow metadata
        r = await self._client.get(f"/repos/{owner}/{repo}/actions/workflows")
        r.raise_for_status()
        data = r.json()
        workflows = data.get("workflows", [])

        files: List[WorkflowFile] = []
        for wf in workflows:
            path = wf.get("path")
            if not path:
                continue
            content_resp = await self._client.get(f"/repos/{owner}/{repo}/contents/{path}")
            content_resp.raise_for_status()
            content_data = content_resp.json()
            # GitHub returns base64 for file content; we keep it simple:
            import base64

            raw = base64.b64decode(content_data.get("content", "")).decode("utf-8", errors="ignore")
            files.append(
                WorkflowFile(
                    id=wf.get("id"),
                    name=wf.get("name"),
                    path=path,
                    raw_yaml=raw,
                )
            )

        return files

    async def close(self) -> None:
        await self._client.aclose()
