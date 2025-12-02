# backend/src/api/routes/rag.py

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_optional_current_user, get_rag_orchestrator
from api.schemas.rag import (
    RAGDebugRequest,
    RAGDebugResponse,
    RAGExplainRequest,
    RAGExplainResponse,
    RAGQueryRequest,
    RAGQueryResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/rag",
    tags=["RAG"],
)


def _build_error_response(message: str) -> RAGQueryResponse:
    """Construct a stable error payload for the caller."""

    return RAGQueryResponse(answer=message, sources=[], intent="error", debug={"error": message})


@router.get("/ping", summary="Lightweight RAG health check")
async def rag_ping() -> dict[str, str]:
    """Lightweight ping endpoint to verify that the RAG router is alive."""

    return {"status": "ok", "message": "RAG router is alive"}


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="Run a RAG query",
    response_description="Answer with citations and diagnostics.",
)
async def rag_query(
    payload: RAGQueryRequest,
    current_user: Any = Depends(get_optional_current_user),
    orchestrator: Any = Depends(get_rag_orchestrator),
) -> RAGQueryResponse:
    """Main endpoint for RAG queries using a pluggable orchestrator."""

    question = (payload.query or "").strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Query must not be empty.",
        )

    org_id = None
    if current_user is not None:
        org_id = getattr(current_user, "org_id", None) or getattr(current_user, "orgId", None)
        if org_id is None and isinstance(current_user, dict):
            org_id = current_user.get("org_id")
    else:
        # No auth provided
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        result = await orchestrator.run(
            org_id=org_id or "unknown-org",
            query=question,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            top_k=payload.top_k,
            metadata=payload.metadata,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAG query failed: %s", exc)
        return _build_error_response("Internal RAG error")

    if isinstance(result, RAGQueryResponse):
        return result

    try:
        return RAGQueryResponse(**result)
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected RAG orchestrator shape")
        return _build_error_response("RAG response was malformed")


@router.post("/explain", response_model=RAGExplainResponse, summary="Explain an answer")
async def rag_explain(payload: RAGExplainRequest) -> RAGExplainResponse:
    """Provide a minimal explanation response."""

    question = (payload.query or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    return RAGExplainResponse(explanation=f"No-op explanation for: {question}", sources=[])


@router.post("/debug", response_model=RAGDebugResponse, summary="Debug a RAG query")
async def rag_debug(payload: RAGDebugRequest) -> RAGDebugResponse:
    """Return lightweight debug information for a query."""

    question = (payload.query or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    return RAGDebugResponse(debug_trace={"message": f"Debug info for: {question}"})
