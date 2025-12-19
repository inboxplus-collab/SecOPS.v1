from __future__ import annotations

import base64
import json
import os
import sys
from threading import Lock
from typing import Any, Dict, Optional, Union

import numpy as np

# Dynamically expose Project A so imports resolve without altering upstream code.
PROJECT_A_PATH = os.path.join(os.path.dirname(__file__), "project_a")
if PROJECT_A_PATH not in sys.path:
    sys.path.insert(0, PROJECT_A_PATH)

from project_a.inference import ProjectAInference  # type: ignore  # noqa: E402
from project_a.preprocess import preprocess_image  # type: ignore  # noqa: E402
from project_a import ProjectAModel, preprocess_image  # type: ignore  # noqa: E402
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
from threading import Lock
from typing import Any, Dict, Optional, Union

import threading
from typing import Any, Dict, Optional, Union

# Dynamically expose ./project_a so legacy imports continue to work.
from threading import Lock
from typing import Any, Dict, Optional, Union

# Minimal change path injection so Project A remains importable after being nested.
PROJECT_A_PATH = Path(__file__).resolve().parent / "project_a"
if str(PROJECT_A_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_A_PATH))

from project_a.inference import ProjectAModel  # noqa: E402

# Ensure Project A remains importable after being nested under ./project_a
PROJECT_A_PATH = os.path.join(os.path.dirname(__file__), "project_a")
if PROJECT_A_PATH not in sys.path:
    sys.path.insert(0, PROJECT_A_PATH)

from project_a.inference import ProjectAModel  # type: ignore  # noqa: E402

ProjectBPayload = Union[str, bytes, Dict[str, Any]]


class ProjectABridge:
    """Adapter that exposes Project A's inference to Project B.

    The bridge keeps Project A importable without modifying its internal
    modules, normalizes the payload shapes expected from Project B, and
    ensures the underlying model is initialized only once to conserve
    memory.
    """Adapter that translates Project B payloads into Project A inputs.

    - Injects ``./project_a`` on ``sys.path`` at runtime.
    - Converts Project B base64/text payloads into the string format Project A expects.
    - Holds a singleton Project A model instance to avoid repeated allocations.
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

    def __new__(cls, *_: Any, **__: Any) -> "ProjectABridge":
    def __new__(cls) -> "AIBridge":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._model = ProjectAInference()
                    cls._instance._model: Optional[ProjectAModel] = None
                    cls._instance._model = ProjectAModel()
        return cls._instance

    def __init__(self) -> None:
        # Initialization handled in __new__ to enforce singleton behavior.
        pass

    def _extract_base64(self, payload: ProjectBPayload) -> str:
                    cls._instance._model: Optional[ProjectAModel] = None
        return cls._instance

    def _get_model(self) -> ProjectAModel:
        if self._model is None:
            self._model = ProjectAModel()

    def _normalize_payload(self, payload: ProjectBPayload) -> str:
        """Convert Project B payloads (base64/plain/dict) into text."""

    def _normalize_payload(self, payload: ProjectBPayload) -> str:
        """Convert Project B payloads to the plain-text format Project A expects."""

        return self._model

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
                base64.b64decode(stripped_payload, validate=True)
                return stripped_payload
            except (ValueError, base64.binascii.Error):
                # Not valid base64; assume raw text for the model.
                return stripped_payload

        if isinstance(payload, dict):
            # Common Project B patterns: {"data": "..."} or {"text": "..."}
            candidate = payload.get("data") or payload.get("text")
            if isinstance(candidate, bytes):
                candidate = candidate.decode("utf-8")
            if isinstance(candidate, str):
                return self._normalize_payload(candidate)
            candidate = payload.get("data") or payload.get("text") or payload.get("payload")
            if isinstance(candidate, (str, bytes)):
                return self._transform_input(candidate)
            raise ValueError("Unsupported dictionary payload structure for Project B input")

        raise TypeError(f"Unsupported payload type: {type(payload)!r}")

    def execute(self, project_b_payload: ProjectBPayload) -> Dict[str, Any]:
        text_input = self._normalize_payload(project_b_payload)
        prediction = self._model.predict(text_input)
        return {"status": "success", "data": prediction}


__all__ = ["ProjectABridge", "ProjectBPayload"]
        """Run Project A inference and return a Project B-friendly response."""

        text_input = self._normalize_payload(project_b_payload)
        prediction = self._model.predict(text_input)
        return {
            "status": "success",
            "input": text_input,
            "data": prediction,
        }


__all__ = ["ProjectABridge", "ProjectBPayload"]
        text_input = self._transform_input(project_b_payload)
        result = self._get_model().predict(text_input)
        return {"status": "success", "input": text_input, "prediction": result}


            for key in ("image", "data", "payload", "body"):
                candidate = payload.get(key)
                if isinstance(candidate, bytes):
                    try:
                        return candidate.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                if isinstance(candidate, str):
                    return candidate
            raise ValueError("Dictionary payload missing a base64-encoded image field")

        raise TypeError(f"Unsupported payload type: {type(payload)!r}")

    def _decode_payload(self, payload: ProjectBPayload) -> np.ndarray:
        base64_string = self._extract_base64(payload)
        try:
            decoded_bytes = base64.b64decode(base64_string, validate=True)
        except (base64.binascii.Error, ValueError):
            decoded_bytes = base64.b64decode(base64_string)

        # If the decoded bytes look like JSON, attempt to reconstruct an array.
        try:
            json_payload = json.loads(decoded_bytes.decode("utf-8"))
            width = int(json_payload.get("width", 0))
            height = int(json_payload.get("height", 0))
            data = json_payload.get("data")
            if isinstance(data, list) and width > 0 and height > 0:
                array = np.array(data, dtype=np.float32).reshape((height, width, -1))
                return array
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            pass

        # Default assumption: Project B sends a raw base64 image.
        return preprocess_image(base64_string)

    def execute(self, project_b_payload: ProjectBPayload) -> Dict[str, Any]:
        """Run Project A inference using Project B's payload format."""

        image_array = self._decode_payload(project_b_payload)
        prediction = self._model.predict(image_array)
        return {"status": "success", "prediction": prediction}


__all__ = ["AIBridge"]
