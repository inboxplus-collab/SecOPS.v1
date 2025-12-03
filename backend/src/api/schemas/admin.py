from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AdminHealthResponse(BaseModel):
    app_name: str
    app_version: str
    environment: str
    status: str
    timestamp: str
    metrics_summary: Dict[str, Any]
    cost_summary: Dict[str, Any]


class AdminReadinessResponse(BaseModel):
    status: str
    llm_configured: bool
    supabase_configured: bool
    vector_store_type: str | None = None
    vector_store_configured: bool
    details: Dict[str, Any] = Field(default_factory=dict)


class AdminMetricsResponse(BaseModel):
    summary: Dict[str, Any]
    today: Dict[str, Any]
    recent_queries: List[Dict[str, Any]]


class AdminCostResponse(BaseModel):
    today: Dict[str, Any]
    month: Dict[str, Any]
    today_over_budget: bool
    month_over_budget: bool


class AdminUser(BaseModel):
    id: str
    email: str
    role: str
    tier: str
    status: str
    mfa_enabled: bool = False
    last_seen: str | None = None


class AdminUserListResponse(BaseModel):
    users: List[AdminUser]


class AdminAuditLog(BaseModel):
    id: str
    actor: str
    action: str
    target: str
    created_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AdminAuditLogResponse(BaseModel):
    items: List[AdminAuditLog]


class AdminAgentActivity(BaseModel):
    id: str
    agent: str
    event: str
    status: str
    created_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AdminAgentActivityResponse(BaseModel):
    items: List[AdminAgentActivity]


class AdminSystemStatus(BaseModel):
    services: Dict[str, str]
    license_usage: Dict[str, Any]
    billing_status: Dict[str, Any]
    uptime_seconds: int
    environment: str


class AdminSystemStatusResponse(BaseModel):
    status: str
    details: AdminSystemStatus
