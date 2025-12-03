from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.src.services.cicd_log_analyzer import cicd_log_analyzer
from services.ai_orchestrator import AIOrchestrator, AnalysisRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["Analysis"])


class AnalysisRunRequest(BaseModel):
    repo_path: Optional[str] = None
    github_repo: Optional[str] = None
    database_url: Optional[str] = None
    database_schema: Optional[Dict[str, Any]] = None
    container_images: List[str] = Field(default_factory=list)
    k8s_enabled: bool = False


class CheckFinding(BaseModel):
    check_id: str
    title: str
    description: str
    severity: str
    metadata: Dict[str, Any]
    detected_at: Any


class AnalysisRunResponse(BaseModel):
    codebase: Dict[str, Any]
    database: Dict[str, Any]
    checks: List[CheckFinding]
    scanners: List[Dict[str, Any]]
    rag_summary: Dict[str, Any]


@router.post(
    "/run",
    response_model=AnalysisRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Run multi-target analysis pipeline",
)
async def run_analysis(request: AnalysisRunRequest) -> AnalysisRunResponse:
    orchestrator = AIOrchestrator()

    try:
        result = await orchestrator.run(
            AnalysisRequest(
                repo_path=request.repo_path,
                github_repo=request.github_repo,
                database_url=request.database_url,
                database_schema=request.database_schema,
                container_images=request.container_images,
                k8s_enabled=request.k8s_enabled,
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Analysis orchestration failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    return AnalysisRunResponse(**result)


@router.get("/ci/github")
async def analyze_github_ci(
    run_id: int | None = Query(None, description="Optional specific run id"),
    branch: str | None = Query(None, description="Optional branch filter"),
):
    """
    Analyze a failing GitHub Actions run (or latest failed run on a branch).
    """

    return await cicd_log_analyzer.analyze_github_run(run_id=run_id, branch=branch)


@router.post("/ci/github/auto-fix")
async def auto_fix_github_workflow(
    workflow_path: str = Query(
        ..., description="Path to workflow file, e.g. .github/workflows/ci.yml"
    ),
    run_id: int | None = Query(None),
    branch: str | None = Query(None),
):
    """
    Automatically propose and apply a fixed GitHub Actions workflow file.
    """

    return await cicd_log_analyzer.auto_fix_github_workflow(
        workflow_path=workflow_path,
        run_id=run_id,
        branch=branch,
    )


@router.get("/ci/jenkins")
async def analyze_jenkins_ci(
    job_name: str = Query(..., description="Jenkins job name")
):
    """
    Analyze the latest failed Jenkins build for a job.
    """

    return await cicd_log_analyzer.analyze_jenkins_job(job_name)
