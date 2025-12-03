from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from api.schemas.docs_chat import DocsChatAnswer, QueryInput
from rag.docs_chat import llm, vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/docs", tags=["Docs"])


@router.post("/chat", response_model=DocsChatAnswer)
async def docs_chat(payload: QueryInput) -> DocsChatAnswer:
    """Query the documentation knowledge base and return an AI answer with citations."""

    query = (payload.text or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query text is required")

    try:
        results = vector_store.search(query)
        answer = llm.answer_with_citations(query, results)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Docs chat failed: %s", exc)
        raise HTTPException(status_code=500, detail="Docs chat unavailable")

    return DocsChatAnswer(**answer)
