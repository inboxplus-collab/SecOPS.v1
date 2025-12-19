"""Bridge layer between Project B and Project A's inference API.

This adapter keeps changes to both codebases minimal by:
- Dynamically adding ``./project_a`` to ``sys.path`` so existing imports work.
- Translating Project B payloads (plain text, base64 strings, or small dicts) into
  the text input expected by Project A.
- Holding a singleton Project A model instance to avoid repeated allocations.
"""

from __future__ import annotations

import base64
import os
import sys
from threading import Lock
from typing import Any, Dict, Optional, Union

# Ensure Project A remains importable after being nested under ./project_a
PROJECT_A_PATH = os.path.join(os.path.dirname(__file__), "project_a")
if PROJECT_A_PATH not in sys.path:
    sys.path.insert(0, PROJECT_A_PATH)

from project_a.model import ProjectAModel  # noqa: E402

ProjectBPayload = Union[str, bytes, Dict[str, Any]]


class ProjectABridge:
    """Adapter that translates Project B payloads into Project A inputs."""

    _instance: Optional["ProjectABridge"] = None
    _lock: Lock = Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "ProjectABridge":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._model = None
        return cls._instance

    def __init__(self) -> None:
        if self._model is None:
            self._model = self._init_model()

    def _init_model(self) -> ProjectAModel:
        return ProjectAModel()

    def _transform_input(self, payload: ProjectBPayload) -> str:
        """Accept Project B formats and convert to Project A's expected string."""

        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")

        if isinstance(payload, str):
            stripped_payload = payload.strip()
            try:
                decoded_bytes = base64.b64decode(stripped_payload, validate=True)
                return decoded_bytes.decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                return stripped_payload

        if isinstance(payload, dict):
            candidate = payload.get("data") or payload.get("text") or payload.get("payload")
            if isinstance(candidate, bytes):
                candidate = candidate.decode("utf-8")
            if isinstance(candidate, str):
                return self._transform_input(candidate)
            raise ValueError("Unsupported dictionary payload structure for Project B input")

        raise TypeError(f"Unsupported payload type: {type(payload)!r}")

    def execute(self, project_b_payload: ProjectBPayload) -> Dict[str, Any]:
        """Run Project A inference using Project B's payload format."""

        text_input = self._transform_input(project_b_payload)
        result = self._model.predict(text_input)
        return {"status": "success", "input": text_input, "prediction": result}


__all__ = ["ProjectABridge", "ProjectBPayload"]
