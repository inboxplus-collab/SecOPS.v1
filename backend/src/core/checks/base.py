from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from api.schemas.issues import (
    IssueDetail,
    IssueLocation,
    IssueResolutionState,
    IssueSeverity,
    IssueStatus,
)


class LoggerLike(Protocol):
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None: ...

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None: ...

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None: ...

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None: ...


@dataclass
class CheckContext:
    """
    Inputs needed by checks to run.

    Includes handles to integrations (GitHub, K8s, CI, scanners, etc.)
    and global configuration.
    """

    org_id: Optional[str]
    settings: Any
    extras: Dict[str, Any] | None = None

    def get_extra(self, key: str, default: Any = None) -> Any:
        if not self.extras:
            return default
        return self.extras.get(key, default)


@dataclass
class CheckIssuePayload:
    """
    Lightweight issue representation produced by checks before persistence.
    """

    id: str
    org_id: str
    check_name: str
    title: str
    severity: IssueSeverity
    status: IssueStatus
    service: Optional[str]
    category: Optional[str]
    tags: List[str]
    location: Optional[IssueLocation]
    source: Optional[str]
    short_description: Optional[str]
    description: Optional[str]
    root_cause: Optional[str]
    impact: Optional[str]
    proposed_fix: Optional[str]
    precautions: Optional[str]
    references: List[str]
    extra: Dict[str, Any]
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        location_payload = self.location.model_dump() if self.location else None
        return {
            "id": self.id,
            "org_id": self.org_id,
            "check_name": self.check_name,
            "title": self.title,
            "severity": self.severity,
            "status": self.status,
            "service": self.service,
            "category": self.category,
            "tags": self.tags,
            "location": location_payload,
            "source": self.source,
            "short_description": self.short_description,
            "description": self.description,
            "root_cause": self.root_cause,
            "impact": self.impact,
            "proposed_fix": self.proposed_fix,
            "precautions": self.precautions,
            "references": self.references,
            "extra": self.extra,
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
        }


@dataclass
class CheckRunResult:
    """
    Output of a check execution returned to the scheduler.
    """

    issues: List[CheckIssuePayload]
    metrics: Dict[str, Any]
    errors: List[str]

    def to_issue_dicts(self) -> List[Dict[str, Any]]:
        return [issue.to_dict() for issue in self.issues]


class BaseCheck(ABC):
    """
    Base class for all SecOpsAI checks.

    Each subclass must implement `run`, returning a CheckRunResult with
    structured issues, metrics, and any non-fatal errors.
    """

    id: str
    name: str
    description: str

    def __init__(self, default_severity: IssueSeverity) -> None:
        self.default_severity = default_severity

    def build_issue(
        self,
        *,
        id: str,
        org_id: str,
        title: str,
        severity: IssueSeverity,
        status: IssueStatus,
        service: Optional[str],
        category: Optional[str],
        tags: List[str],
        location: Optional[IssueLocation],
        source: Optional[str],
        short_description: Optional[str],
        description: Optional[str],
        root_cause: Optional[str],
        impact: Optional[str],
        proposed_fix: Optional[str],
        precautions: Optional[str],
        references: List[str],
        extra: Dict[str, Any],
        first_seen_at: Optional[datetime] = None,
        last_seen_at: Optional[datetime] = None,
    ) -> CheckIssuePayload:
        return CheckIssuePayload(
            id=id,
            org_id=org_id,
            check_name=self.id,
            title=title,
            severity=severity,
            status=status,
            service=service,
            category=category,
            tags=tags,
            location=location,
            source=source,
            short_description=short_description,
            description=description,
            root_cause=root_cause,
            impact=impact,
            proposed_fix=proposed_fix,
            precautions=precautions,
            references=references,
            extra=extra,
            first_seen_at=first_seen_at,
            last_seen_at=last_seen_at,
        )

    @abstractmethod
    async def run(self, context: CheckContext, logger: LoggerLike) -> CheckRunResult:
        ...
