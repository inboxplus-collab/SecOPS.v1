"""Bridge layer between Project B and Project A."""

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

from project_a import ProjectAModel  # type: ignore  # noqa: E402
from project_a.model import SecurityInferenceModel  # noqa: E402

ProjectBPayload = Union[str, bytes, Dict[str, Any]]


class AIBridge:
    """Singleton bridge that adapts Project B payloads for Project A."""

    _instance: Optional["AIBridge"] = None
    _lock: Lock = Lock()

    def __new__(cls) -> "AIBridge":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._model = None
        return cls._instance

    def __init__(self) -> None:
        if self._model is None:
            self._model = ProjectAModel()

    @classmethod
    def get_instance(cls) -> "AIBridge":
        return cls()

    def _normalize_payload(self, payload: ProjectBPayload) -> str:
        """Convert Project B inputs into the raw text expected by Project A."""

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
            if candidate is None:
                raise ValueError("Dictionary payload must include 'data' or 'text'")
            return self._normalize_payload(candidate)

        raise TypeError(f"Unsupported payload type: {type(payload)!r}")

    def predict(self, project_b_payload: ProjectBPayload) -> Dict[str, Any]:
        """Run Project A inference using Project B's payload format."""

        text_input = self._normalize_payload(project_b_payload)
        prediction = self._model.predict(text_input)
        return {"status": "success", "input": text_input, "prediction": prediction}


__all__ = ["AIBridge"]
    def execute(self, project_b_payload: ProjectBPayload) -> Dict[str, Any]:
        text_input = self._transform_input(project_b_payload)
        result = self._model.predict(text_input)
        return {
            "status": "success",
            "input": text_input,
            "prediction": result,
        }


__all__ = ["ProjectABridge"]
