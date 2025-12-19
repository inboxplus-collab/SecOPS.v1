"""Bridge adapter connecting Project B to Project A's inference API."""
"""Bridge layer between Project B and Project A's inference API.

This adapter keeps changes to both codebases minimal by:
- Dynamically adding ``./project_a`` to ``sys.path`` so existing imports work.
- Translating Project B payloads (plain text, base64 strings, or small dicts) into
  the text input expected by Project A.
- Holding a singleton Project A model instance to avoid repeated allocations.
"""
"""Bridge layer between Project B and Project A."""

from __future__ import annotations

import base64
import os
import sys
import threading
from typing import Any, Dict, Optional, Union

# Dynamically expose ./project_a so legacy imports continue to work.
from threading import Lock
from typing import Any, Dict, Optional, Union

# Ensure Project A remains importable after being nested under ./project_a
PROJECT_A_PATH = os.path.join(os.path.dirname(__file__), "project_a")
if PROJECT_A_PATH not in sys.path:
    sys.path.insert(0, PROJECT_A_PATH)

from project_a.inference import ProjectAModel  # type: ignore  # noqa: E402

ProjectBPayload = Union[str, bytes, Dict[str, Any]]


class ProjectABridge:
    """Adapter class that translates Project B payloads for Project A.

    Responsibilities:
    - Adds ``./project_a`` to ``sys.path`` so imports inside Project A resolve.
    - Converts common Project B payload formats (base64 strings, bytes, dicts)
      into the raw text expected by Project A's ``ProjectAModel``.
    - Maintains a singleton model instance to minimize memory usage.
    - Returns predictions in a Project B friendly dictionary structure.
    """

    _instance: Optional["ProjectABridge"] = None
    _lock = threading.Lock()
from project_a.model import ProjectAModel  # noqa: E402
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
                    cls._instance._model: Optional[ProjectAModel] = None
        return cls._instance

    def __init__(self) -> None:
        if self._model is None:
            self._model = ProjectAModel()

    def _decode_payload(self, payload: ProjectBPayload) -> str:
        """Normalize Project B payloads into the text Project A expects."""

            self._model = self._init_model()

    def _init_model(self) -> ProjectAModel:
        return ProjectAModel()
            self._model = ProjectAModel()

    @classmethod
    def get_instance(cls) -> "AIBridge":
        return cls()

    def _normalize_payload(self, payload: ProjectBPayload) -> str:
        """Convert Project B inputs into the raw text expected by Project A."""

            self._model = SecurityInferenceModel()

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
                return self._decode_payload(candidate)
            raise ValueError("Unsupported dictionary payload structure for Project B input")

        raise TypeError(f"Unsupported payload type: {type(payload)!r}")

    def execute(self, project_b_payload: ProjectBPayload) -> Dict[str, Any]:
        """Run Project A inference and return a Project B friendly response."""

        normalized_text = self._decode_payload(project_b_payload)
        prediction = self._model.predict(normalized_text)
        return {
            "status": "success",
            "input": normalized_text,
            "prediction": prediction,
        """Run Project A inference using Project B's payload format."""

        text_input = self._transform_input(project_b_payload)
        result = self._model.predict(text_input)
        return {"status": "success", "input": text_input, "prediction": result}


__all__ = ["ProjectABridge", "ProjectBPayload"]
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
