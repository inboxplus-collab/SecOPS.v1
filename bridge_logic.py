from __future__ import annotations

import base64
import sys
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Union

# Dynamically expose Project A (./project_a) so its imports resolve without altering upstream code.
PROJECT_A_PATH = Path(__file__).resolve().parent / "project_a"
if str(PROJECT_A_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_A_PATH))

from project_a.model import SecurityInferenceModel  # noqa: E402

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
            self._model = SecurityInferenceModel()

    def _transform_input(self, payload: ProjectBPayload) -> str:
        """Accept Project B data formats and convert to Project A's expected string."""
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")

        if isinstance(payload, str):
            try:
                decoded = base64.b64decode(payload, validate=True)
                return decoded.decode("utf-8")
            except Exception:
                return payload

        if isinstance(payload, dict):
            candidate = payload.get("data") or payload.get("text")
            if isinstance(candidate, bytes):
                candidate = candidate.decode("utf-8")
            if isinstance(candidate, str):
                return self._transform_input(candidate)
            raise ValueError("Unsupported dictionary payload structure for Project B input")

        raise TypeError(f"Unsupported payload type: {type(payload)!r}")

    def execute(self, project_b_payload: ProjectBPayload) -> Dict[str, Any]:
        text_input = self._transform_input(project_b_payload)
        result = self._model.predict(text_input)
        return {
            "status": "success",
            "input": text_input,
            "prediction": result,
        }


__all__ = ["ProjectABridge"]
