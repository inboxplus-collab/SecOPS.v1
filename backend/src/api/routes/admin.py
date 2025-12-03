# backend/src/api/routes/admin.py

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query

from api.deps import require_admin
from api.schemas.admin import (
    AdminAgentActivity,
    AdminAgentActivityResponse,
    AdminAuditLog,
    AdminAuditLogResponse,
    AdminCostResponse,
    AdminHealthResponse,
    AdminMetricsResponse,
    AdminReadinessResponse,
    AdminSystemStatus,
    AdminSystemStatusResponse,
    AdminUser,
    AdminUserListResponse,
)
from services.billing import BillingDB
from services.metrics_collector import metrics_collector
from services.cost_tracker import cost_tracker
from utils.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/admin",
    tags=["Admin"],
)


@router.get(
    "/health",
    response_model=AdminHealthResponse,
    summary="Application health overview",
    response_description="High-level health info plus basic metrics snapshot.",
)
async def get_health() -> AdminHealthResponse:
    """
    Lightweight health endpoint for dashboards and uptime checks.

    Returns:
    - app metadata (name, version, environment)
    - current UTC timestamp
    - basic metrics summary (queries, latency, cost)
    """
    metrics_summary = metrics_collector.get_summary()
    cost_stats = cost_tracker.get_stats()

    return AdminHealthResponse(
        app_name=settings.APP_NAME,
        app_version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        status="ok",
        timestamp=datetime.utcnow().isoformat(),
        metrics_summary=metrics_summary,
        cost_summary=cost_stats,
    )


@router.get(
    "/readiness",
    response_model=AdminReadinessResponse,
    summary="Readiness & configuration check",
    response_description="Indicates whether backend is ready for production traffic.",
)
async def get_readiness() -> AdminReadinessResponse:
    """
    Readiness probe for deployment / load balancers.

    It does NOT invoke the full RAG engine, but checks:
    - LLM credentials present
    - Supabase / storage configuration present (if enabled)
    - Vector store configuration
    """
    # LLM configuration
    llm_configured = bool(
        settings.EMERGENT_LLM_KEY
        or settings.OPENAI_API_KEY
        or settings.ANTHROPIC_API_KEY
        or settings.GEMINI_API_KEY
    )

    # Supabase configuration
    supabase_configured = bool(settings.SUPABASE_URL and settings.SUPABASE_KEY)

    # Vector store configuration
    vector_store_type = settings.VECTOR_STORE_TYPE
    vector_store_configured = vector_store_type in {"mock", "qdrant", "supabase_pgvector"}

    all_ok = llm_configured and vector_store_configured
    status = "ready" if all_ok else "degraded"

    return AdminReadinessResponse(
        status=status,
        llm_configured=llm_configured,
        supabase_configured=supabase_configured,
        vector_store_type=vector_store_type,
        vector_store_configured=vector_store_configured,
        details={
            "env": settings.ENVIRONMENT,
            "debug_mode": settings.DEBUG,
        },
    )


@router.get(
    "/metrics",
    response_model=AdminMetricsResponse,
    summary="Metrics & recent queries",
    response_description="Aggregated metrics plus recent query records.",
)
async def get_metrics(
    limit: int = Query(10, ge=1, le=100, description="Number of recent queries to return"),
) -> AdminMetricsResponse:
    """
    Return aggregated metrics and recent query records.

    Useful for admin console, internal monitoring, and debugging.
    """
    summary = metrics_collector.get_summary()
    today = metrics_collector.get_today_summary()
    recent = metrics_collector.get_recent_queries(limit=limit)

    return AdminMetricsResponse(
        summary=summary,
        today=today,
        recent_queries=recent,
    )


@router.get(
    "/cost",
    response_model=AdminCostResponse,
    summary="LLM cost & budget stats",
    response_description="Current spending vs daily/monthly budgets.",
)
async def get_cost() -> AdminCostResponse:
    """
    Surface LLM usage cost and budget status.

    Reads directly from the global CostTracker.
    """
    stats = cost_tracker.get_stats()

    # Derive simple flags for dashboards
    today = stats.get("today", {})
    month = stats.get("month", {})

    today_over = today.get("cost", 0.0) > today.get("budget", 0.0)
    month_over = month.get("cost", 0.0) > month.get("budget", 0.0)

    return AdminCostResponse(
        today=today,
        month=month,
        today_over_budget=today_over,
        month_over_budget=month_over,
    )


@router.get(
    "/users",
    response_model=AdminUserListResponse,
    dependencies=[Depends(require_admin)],
    summary="List all users with subscription tier",
)
async def list_users() -> AdminUserListResponse:
    """Return a synthetic list of users enriched with billing tiers."""

    # In a real deployment this would query the identity provider or database.
    sample_users = [
        AdminUser(
            id="user-1",
            email="founder@secops.ai",
            role="admin",
            tier="enterprise",
            status="active",
            mfa_enabled=True,
            last_seen=datetime.utcnow().isoformat(),
        ),
        AdminUser(
            id="user-2",
            email="engineer@secops.ai",
            role="member",
            tier="pro",
            status="active",
            mfa_enabled=False,
            last_seen=(datetime.utcnow() - timedelta(hours=2)).isoformat(),
        ),
        AdminUser(
            id="user-3",
            email="guest@secops.ai",
            role="viewer",
            tier=BillingDB.get_status("guest@secops.ai").get("tier", "free"),
            status="invited",
            mfa_enabled=False,
            last_seen=None,
        ),
    ]

    return AdminUserListResponse(users=sample_users)


@router.get(
    "/audit",
    response_model=AdminAuditLogResponse,
    dependencies=[Depends(require_admin)],
    summary="Audit log events for SOC2",
)
async def get_audit_log(limit: int = Query(20, ge=1, le=100)) -> AdminAuditLogResponse:
    """Return recent audit events for admin dashboards."""

    now = datetime.utcnow()
    items = [
        AdminAuditLog(
            id=f"audit-{i}",
            actor="system",
            action="policy.update" if i % 2 == 0 else "scan.run",
            target=f"service-{i}",
            created_at=(now - timedelta(minutes=i * 5)).isoformat(),
            metadata={"severity": "info", "source": "SecOpsAI"},
        )
        for i in range(limit)
    ]
    return AdminAuditLogResponse(items=items)


@router.get(
    "/agent-activity",
    response_model=AdminAgentActivityResponse,
    dependencies=[Depends(require_admin)],
    summary="Recent autonomous agent actions",
)
async def get_agent_activity(limit: int = Query(15, ge=1, le=50)) -> AdminAgentActivityResponse:
    """List synthetic agent events to power the admin UI."""

    now = datetime.utcnow()
    events = [
        AdminAgentActivity(
            id=f"agent-{i}",
            agent="secops-auto-fix",
            event="patch_applied" if i % 3 == 0 else "scan_completed",
            status="succeeded" if i % 4 != 0 else "pending",
            created_at=(now - timedelta(minutes=i * 3)).isoformat(),
            metadata={"target": f"service-{i}", "cluster": "prod-us"},
        )
        for i in range(limit)
    ]
    return AdminAgentActivityResponse(items=events)


@router.get(
    "/system",
    response_model=AdminSystemStatusResponse,
    dependencies=[Depends(require_admin)],
    summary="System health, license, and billing status",
)
async def get_system_status() -> AdminSystemStatusResponse:
    """Expose holistic system state for the admin dashboard."""

    billing_snapshot = BillingDB.get_status("admin@secops.ai")
    status = AdminSystemStatus(
        services={
            "api": "healthy",
            "rag": "healthy",
            "scheduler": "degraded" if settings.DEBUG else "healthy",
        },
        license_usage={"seats": 25, "used": 18, "percent": 72},
        billing_status=billing_snapshot,
        uptime_seconds=3600 * 48,
        environment=settings.ENVIRONMENT,
    )

    return AdminSystemStatusResponse(status="ok", details=status)
