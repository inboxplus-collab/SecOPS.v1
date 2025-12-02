# backend/src/extensions/feature_store.py

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging


@dataclass
class FeatureEvent:
    """
    A single observation / event the system can learn from.

    Examples:
      - a check finds an issue and the user marks it as "real" or "false_positive"
      - an issue is resolved quickly vs. ignored
      - a specific check configuration leads to more useful findings

    This is intentionally generic: you control the meaning of `features`.
    """

    id: str
    kind: str  # e.g. "issue_feedback", "check_result"
    timestamp: datetime
    features: Dict[str, Any]
    label: Optional[float] = None  # e.g. reward score: +1, 0, -1

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureEvent":
        return cls(
            id=str(data["id"]),
            kind=str(data["kind"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            features=dict(data.get("features", {})),
            label=data.get("label"),
        )


class FeatureStore:
    """
    Very small feature store that keeps a rolling window of events in memory,
    and optionally persists them to a JSONL file for offline analysis or
    training.

    This is deliberately simple and dependency-free. In the future, you can
    swap it out for a real feature store (e.g. Feast, Supabase tables, etc.)
    without changing the rest of the extension APIs.
    """

    def __init__(self, path: Optional[Path] = None, max_events: int = 10_000) -> None:
        self._logger = logging.getLogger(__name__)
        self._path = path
        self._max_events = max_events
        self._events: List[FeatureEvent] = []
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self._events.append(FeatureEvent.from_dict(data))
        except Exception:
            # For an optional extension module, we fail soft.
            # You can add logging here if desired.
            self._events = []

    def _persist_event(self, event: FeatureEvent) -> None:
        if self._path is None:
            return
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception as exc:
            # Optional persistence – log and continue without crashing the app.
            self._logger.warning("FeatureStore persistence failed: %s", exc)

    @property
    def events(self) -> List[FeatureEvent]:
        return list(self._events)

    def add_event(self, event: FeatureEvent) -> None:
        """
        Add a new event to the store and persist it if configured.
        """
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events :]
        self._persist_event(event)

    def get_recent(self, limit: int = 100, kind: Optional[str] = None) -> List[FeatureEvent]:
        """
        Return recent events, optionally filtered by kind.
        """
        items = self._events
        if kind is not None:
            items = [e for e in items if e.kind == kind]
        return items[-limit:]
