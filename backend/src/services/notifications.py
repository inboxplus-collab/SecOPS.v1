from __future__ import annotations

import os
from typing import Optional

import httpx


class NotificationService:
    """
    Placeholder notification system.

    For MVP: supports Slack webhook if SLACK_WEBHOOK_URL is set.
    """

    def __init__(self) -> None:
        self._slack_url: Optional[str] = os.getenv("SLACK_WEBHOOK_URL")

    async def notify_text(self, text: str) -> None:
        if not self._slack_url:
            return
        async with httpx.AsyncClient() as client:
            await client.post(self._slack_url, json={"text": text})
