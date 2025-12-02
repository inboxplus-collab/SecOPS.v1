from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class IssueSeverity(str, Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"
    info = "info"


class IssueStatus(str, Enum):
    open = "open"
    in_progress = "in_progress"
    resolved = "resolved"
    suppressed = "suppressed"


class IssueResolutionState(str, Enum):
    resolved = "resolved"
    suppressed = "suppressed"
    acknowledged = "acknowledged"


# ---------------------------------------------------------------------------
# Location + response DTOs
# ---------------------------------------------------------------------------


class IssueLocation(BaseModel):
    kind: Optional[str] = None
    repo: Optional[str] = None
    file_path: Optional[str] = None
    line: Optional[int] = None
    cluster: Optional[str] = None
    namespace: Optional[str] = None
    resource_kind: Optional[str] = None
    resource_name: Optional[str] = None
    environment: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IssueSummary(BaseModel):
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
    first_seen_at: Optional[datetime]
    last_seen_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    short_description: Optional[str] = None


class IssueDetail(IssueSummary):
    description: Optional[str] = None
    root_cause: Optional[str] = None
    impact: Optional[str] = None
    proposed_fix: Optional[str] = None
    precautions: Optional[str] = None
    references: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_state: Optional[IssueResolutionState] = None
    resolution_note: Optional[str] = None


# ---------------------------------------------------------------------------
# CRUD payloads
# ---------------------------------------------------------------------------


class IssueBase(BaseModel):
    title: str
    description: Optional[str] = None
    severity: IssueSeverity = Field(
        IssueSeverity.medium, description="critical|high|medium|low|info"
    )
    status: IssueStatus = Field(
        IssueStatus.open, description="open|in_progress|resolved|ignored"
    )
    source: Optional[str] = Field(None, description="github|k8s|ci|scanner|manual")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IssueCreate(IssueBase):
    org_id: str
    check_name: str
    service: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    location: Optional[IssueLocation] = None
    short_description: Optional[str] = None
    root_cause: Optional[str] = None
    impact: Optional[str] = None
    proposed_fix: Optional[str] = None
    precautions: Optional[str] = None
    references: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolution_state: Optional[IssueResolutionState] = None
    resolution_note: Optional[str] = None
    resolved_by: Optional[str] = None


class IssueUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    severity: Optional[IssueSeverity] = None
    status: Optional[IssueStatus] = None
    metadata: Optional[Dict[str, Any]] = None


class IssueRead(IssueBase):
    id: str
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class IssueListResponse(BaseModel):
    items: List[IssueRead]
    total: int
    page: int
    page_size: int
