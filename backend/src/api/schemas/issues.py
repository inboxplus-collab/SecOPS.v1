from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IssueBase(BaseModel):
    title: str
    description: Optional[str] = None
    severity: str = Field("medium", description="critical|high|medium|low|info")
    status: str = Field("open", description="open|in_progress|resolved|ignored")
    source: Optional[str] = Field(None, description="github|k8s|ci|scanner|manual")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IssueCreate(IssueBase):
    pass


class IssueUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    severity: Optional[str] = None
    status: Optional[str] = None
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
