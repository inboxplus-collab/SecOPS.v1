from typing import Any, Dict, List, Optional

import httpx


class GitLabConnector:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"PRIVATE-TOKEN": token}

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None):
        url = f"{self.base_url}/api/v4/{path.lstrip('/')}"
        async with httpx.AsyncClient(headers=self.headers) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    async def list_projects(self) -> List[Dict[str, Any]]:
        return await self._get("projects")

    async def list_commits(self, project_id: str) -> List[Dict[str, Any]]:
        return await self._get(f"projects/{project_id}/repository/commits")

    async def get_pipeline_jobs(self, project_id: str, pipeline_id: str) -> List[Dict[str, Any]]:
        return await self._get(f"projects/{project_id}/pipelines/{pipeline_id}/jobs")
