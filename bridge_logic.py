from __future__ import annotations

import base64
import sys
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Union

# Minimal change path injection so Project A remains importable after being nested.
PROJECT_A_PATH = Path(__file__).resolve().parent / "project_a"
if str(PROJECT_A_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_A_PATH))

from project_a.inference import ProjectAModel  # noqa: E402

ProjectBPayload = Union[str, bytes, Dict[str, Any]]


class ProjectABridge:
    """Adapter that exposes Project A's inference to Project B.

    The bridge keeps Project A importable without modifying its internal
    modules, normalizes the payload shapes expected from Project B, and
    ensures the underlying model is initialized only once to conserve
    memory.
    """

    _instance: Optional["ProjectABridge"] = None
    _lock: Lock = Lock()

    def __new__(cls, *_: Any, **__: Any) -> "ProjectABridge":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._model = None
        return cls._instance

    def __init__(self) -> None:
        if self._model is None:
            self._model = ProjectAModel()

    def _normalize_payload(self, payload: ProjectBPayload) -> str:
        """Convert Project B payloads (base64/plain/dict) into text."""
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")

        if isinstance(payload, str):
            try:
                decoded = base64.b64decode(payload, validate=True)
                return decoded.decode("utf-8")
            except Exception:
                return payload

        if isinstance(payload, dict):
            # Common Project B patterns: {"data": "..."} or {"text": "..."}
            candidate = payload.get("data") or payload.get("text")
            if isinstance(candidate, bytes):
                candidate = candidate.decode("utf-8")
            if isinstance(candidate, str):
                return self._normalize_payload(candidate)
            raise ValueError("Unsupported dictionary payload structure for Project B input")

        raise TypeError(f"Unsupported payload type: {type(payload)!r}")

    def execute(self, project_b_payload: ProjectBPayload) -> Dict[str, Any]:
        text_input = self._normalize_payload(project_b_payload)
        prediction = self._model.predict(text_input)
        return {"status": "success", "data": prediction}


__all__ = ["ProjectABridge", "ProjectBPayload"]
