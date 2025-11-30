from __future__ import annotations

from typing import Any, Dict, List


class MetricsCollector:
    """
    Very small in-process metrics collector.
    For serious usage, rely on Prometheus. This can still be used
    to feed RAG or UI stats.
    """

    def __init__(self) -> None:
        self._counters: Dict[str, int] = {}
        self._recent_queries: List[Dict[str, Any]] = []

    def inc(self, name: str, amount: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + amount

    def record_query(
        self,
        query_id: str,
        duration: float,
        tokens: int,
        cost: float,
        intent: str,
        status: str,
        citations_count: int,
        trust_score: float = 0.0,
    ) -> None:
        self._recent_queries.append(
            {
                "id": query_id,
                "duration": duration,
                "tokens": tokens,
                "cost": cost,
                "intent": intent,
                "status": status,
                "citations_count": citations_count,
                "trust_score": trust_score,
            }
        )
        # Keep recent list bounded
        self._recent_queries = self._recent_queries[-100:]
        self.inc("queries_total")

    def snapshot(self) -> Dict[str, int]:
        return dict(self._counters)

    def get_summary(self) -> Dict[str, Any]:
        return {"counters": self.snapshot(), "recent_queries": len(self._recent_queries)}

    def get_today_summary(self) -> Dict[str, Any]:
        # Simplified placeholder
        return {"queries": len(self._recent_queries)}

    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        return list(self._recent_queries[-limit:])


metrics_collector = MetricsCollector()
