from __future__ import annotations

from datetime import date
from typing import List

from pydantic import BaseModel


class PlatformSummary(BaseModel):
    status: str
    issues_open: int
    issues_critical: int
    checks_failing: int
    checks_total: int
    monthly_cost_estimate_usd: float


class IssuesTrendPoint(BaseModel):
    date: date
    open: int
    resolved: int
    new: int


class CheckStatusSlice(BaseModel):
    status: str
    count: int


class CostBreakdownItem(BaseModel):
    category: str  # e.g. "compute", "storage", "network", "llm_api"
    monthly_cost_usd: float


class IssuesTrendResponse(BaseModel):
    points: List[IssuesTrendPoint]


class CheckStatusResponse(BaseModel):
    slices: List[CheckStatusSlice]


class CostBreakdownResponse(BaseModel):
    items: List[CostBreakdownItem]
