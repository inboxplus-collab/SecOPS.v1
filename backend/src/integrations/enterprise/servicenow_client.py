from typing import Any, Dict, List, Optional

import httpx


class ServiceNowConnector:
    def __init__(self, instance_url: str, username: str, password: str):
        self.instance_url = instance_url.rstrip("/")
        self.auth = (username, password)

    async def _request(
        self, method: str, path: str, params: Optional[Dict[str, Any]] = None
    ):
        url = f"{self.instance_url}/{path.lstrip('/')}"
        async with httpx.AsyncClient(auth=self.auth) as client:
            response = await client.request(method, url, params=params)
            response.raise_for_status()
            return response.json()

    async def list_incidents(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {"sysparm_query": query} if query else None
        data = await self._request("GET", "api/now/table/incident", params=params)
        return data.get("result", [])

    async def get_configuration_items(self) -> List[Dict[str, Any]]:
        data = await self._request("GET", "api/now/table/cmdb_ci")
        return data.get("result", [])
