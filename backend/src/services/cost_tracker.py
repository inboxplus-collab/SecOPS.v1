from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class CostCounter:
    category: str
    monthly_usd: float = 0.0


class CostTracker:
    """
    Tracks rough monthly costs for categories like:
      - compute, storage, network, llm_api
    This is intentionally approximate – you can wire real billing later.
    """

    def __init__(self) -> None:
        self._items: Dict[str, CostCounter] = {}

    def add_cost(self, category: str, amount: float) -> None:
        c = self._items.get(category) or CostCounter(category=category, monthly_usd=0.0)
        c.monthly_usd += amount
        self._items[category] = c

    def breakdown(self) -> Dict[str, float]:
        return {k: v.monthly_usd for k, v in self._items.items()}

    def record_cost(self, provider: str, input_tokens: int, output_tokens: int, model: str | None = None) -> float:
        # Very rough estimation placeholder
        cost = (input_tokens + output_tokens) * 0.000001
        self.add_cost("llm_api", cost)
        return cost

    def get_stats(self) -> Dict[str, Any]:
        return {"monthly_usd": self.breakdown()}


cost_tracker = CostTracker()
