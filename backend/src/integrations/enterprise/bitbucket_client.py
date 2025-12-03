from typing import Any, Dict, List, Optional, Tuple

import httpx


class BitbucketConnector:
    def __init__(self, workspace: str, username: str, app_password: str):
        self.workspace = workspace
        self.auth: Tuple[str, str] = (username, app_password)

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None):
        url = f"https://api.bitbucket.org/2.0/{path.lstrip('/')}"
        async with httpx.AsyncClient(auth=self.auth) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    async def list_repositories(self) -> List[Dict[str, Any]]:
        data = await self._get(f"repositories/{self.workspace}")
        return data.get("values", [])

    async def list_commits(self, repo_slug: str) -> List[Dict[str, Any]]:
        data = await self._get(f"repositories/{self.workspace}/{repo_slug}/commits")
        return data.get("values", [])
