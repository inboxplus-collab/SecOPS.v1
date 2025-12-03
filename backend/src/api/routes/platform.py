# backend/src/api/routes/platform.py

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, Query

from api.deps import get_current_user
from backend.src.extensions.k8s_healer.healer import k8s_healer
from extensions.db_repair import db_repair_agent

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/platform",
    tags=["Platform"],
)


@router.get("/status", summary="Lightweight platform status")
async def platform_status(current_user: Any = Depends(get_current_user)) -> Dict[str, Any]:
    """Minimal platform status endpoint used by tests."""

    org_id = getattr(current_user, "org_id", None)
    if org_id is None and isinstance(current_user, dict):
        org_id = current_user.get("org_id")

    return {
        "status": "ok",
        "org_id": org_id or "unknown-org",
        "environment": "test",
    }


@router.post("/k8s/heal", summary="Analyze and optionally heal k8s workloads")
async def k8s_heal(
    apply: bool = Query(
        False, description="Apply patches (true) or perform a dry-run (false)"
    ),
    current_user: Any = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Run Kubernetes misconfiguration analysis and optionally apply auto-heal patches.

    - If apply=false → returns issues + suggested patches (does NOT change cluster)
    - If apply=true  → applies patches via Kubernetes API
    """

    # current_user is resolved for auth; unused otherwise
    _ = current_user
    result = await k8s_healer.heal(apply=apply)
@router.post("/database/repair", summary="Run DB auto-repair checks")
async def repair_db(autofix: bool = False, current_user: Any = Depends(get_current_user)) -> Dict[str, Any]:
    """Run database health checks and optionally apply safe fixes."""

    result = await db_repair_agent.run_full_check()

    if autofix:
        await db_repair_agent.apply_patches(result["patches"])

    return result
