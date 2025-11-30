from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class CheckContext:
    """
    Inputs needed by checks to run.

    Includes handles to integrations (GitHub, K8s, CI, scanners, etc.)
    and global configuration.
    """

    org_id: Optional[str] = None
    project_id: Optional[str] = None
    extra: Dict[str, Any] = None


@dataclass
class CheckResult:
    """
    Outcome of a single check.
    """

    check_id: str
    title: str
    description: str
    severity: str  # critical|high|medium|low|info
    metadata: Dict[str, Any]
    detected_at: datetime


class BaseCheck(ABC):
    """
    Base class for all SecOpsAI checks.

    Each subclass must implement `run`, returning zero or more CheckResult
    objects ready to be converted into Issues.
    """

    id: str
    name: str
    description: str

    @abstractmethod
    async def run(self, ctx: CheckContext) -> List[CheckResult]:
        raise NotImplementedError
