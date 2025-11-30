from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RagQueryPayload(BaseModel):
    question: str = Field(..., min_length=3)
    intent: Optional[str] = Field(None, description="Optional hint: 'security', 'cost', 'devops', etc.")
    context: Dict[str, Any] = Field(default_factory=dict)
    debug: bool = False


class RagCitation(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    snippet: Optional[str] = None
    source_type: Optional[str] = Field(None, description="github|k8s|ci|scanner|doc|web")


class RagQueryResponse(BaseModel):
    answer: str
    intent: str
    mode: str  # e.g. "rag", "direct_llm", "search_only"
    citations: List[RagCitation] = Field(default_factory=list)
    latency_ms: Optional[int] = None
    status: str = "ok"
    error_message: Optional[str] = None
