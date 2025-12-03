# backend/src/integrations/ci/jenkins.py

"""
Jenkins CI integration.

Capabilities:
- List recent builds for a job
- Find latest failed build
- Fetch console log text

Relies on:
- JENKINS_URL (e.g. "https://jenkins.example.com")
- JENKINS_USER
- JENKINS_API_TOKEN
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
import os
from urllib.parse import urljoin


class JenkinsClient:
    def __init__(self):
        self.base_url = os.getenv("JENKINS_URL")
        self.user = os.getenv("JENKINS_USER")
        self.token = os.getenv("JENKINS_API_TOKEN")

        if not self.base_url:
            raise RuntimeError("JENKINS_URL must be set to use JenkinsClient.")

        self.auth = None
        if self.user and self.token:
            self.auth = (self.user, self.token)

    async def list_builds(self, job: str, max_builds: int = 10) -> List[Dict[str, Any]]:
        """
        Uses Jenkins JSON API to list recent builds for a job.
        """
        job_path = f"/job/{job}/api/json?depth=1"
        url = urljoin(self.base_url, job_path)

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, auth=self.auth)
            resp.raise_for_status()
            data = resp.json()

        builds = data.get("builds", [])
        return builds[:max_builds]

    async def get_latest_failed_build(
        self, job: str, max_builds: int = 10
    ) -> Optional[Dict[str, Any]]:
        builds = await self.list_builds(job, max_builds=max_builds)
        for b in builds:
            if b.get("result") == "FAILURE":
                return b
        return None

    async def get_console_log(self, build_url: str) -> str:
        """
        Fetches the console output text for a build.
        """
        url = urljoin(build_url, "consoleText")

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(url, auth=self.auth)
            resp.raise_for_status()
            return resp.text


jenkins_client = JenkinsClient()
