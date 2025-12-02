from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RAGQueryRequest(BaseModel):
    """Request payload for RAG queries."""

    query: str = Field(..., min_length=3, description="User question to answer")
    intent: Optional[str] = Field(None, description="Optional hint: security|cost|devops")
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def question(self) -> str:
        """Compatibility accessor for older naming."""

        return self.query


class RAGQueryResponse(BaseModel):
    """Normalized RAG response returned by the API layer."""

    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    usage: Optional[Dict[str, Any]] = None
    intent: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None


class RAGExplainRequest(BaseModel):
    query: str = Field(..., min_length=3)
    context: Dict[str, Any] = Field(default_factory=dict)


class RAGExplainResponse(BaseModel):
    explanation: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)


class RAGDebugRequest(BaseModel):
    query: str = Field(..., min_length=3)


class RAGDebugResponse(BaseModel):
    debug_trace: Dict[str, Any]
