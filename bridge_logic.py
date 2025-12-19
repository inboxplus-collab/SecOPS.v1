import base64
import os
import sys
import threading
from typing import Any, Dict

project_a_path = os.path.join(os.path.dirname(__file__), "project_a")
if project_a_path not in sys.path:
    sys.path.append(project_a_path)

from project_a.model import SecurityInferenceModel


class ProjectABridge:
    """Adapter that exposes Project A's inference to Project B."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> "ProjectABridge":
from __future__ import annotations

import base64
import sys
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Union

# Dynamically expose Project A (./project_a) so its imports resolve without altering upstream code.
PROJECT_A_PATH = Path(__file__).resolve().parent / "project_a"
if str(PROJECT_A_PATH) not in sys.path:
    sys.path.append(str(PROJECT_A_PATH))

from project_a.inference import ProjectAModel  # type: ignore  # noqa: E402


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

    def _decode_base64_input(self, encoded_payload: str) -> str:
        decoded_bytes = base64.b64decode(encoded_payload)
        return decoded_bytes.decode("utf-8")

    def execute(self, encoded_payload: str) -> Dict[str, Any]:
        """Decode Project B's payload and run Project A's inference."""
        normalized_payload = self._decode_base64_input(encoded_payload)
        prediction = self._model.predict(normalized_payload)
        return {
            "status": "success",
            "input": normalized_payload,
            "prediction": prediction,
        }


__all__ = ["ProjectABridge"]
                    cls._instance._init_model()
        return cls._instance

    def _init_model(self) -> None:
        self.model = ProjectAModel()

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
        result = self.model.predict(text_input)
"""Bridge layer between Project B (FastAPI app) and Project A (AI model).

The bridge dynamically exposes Project A's inference API while keeping
changes to both codebases minimal.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import base64
import sys
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict

from PIL import Image

PROJECT_A_PATH = "./project_a"
if PROJECT_A_PATH not in sys.path:
    sys.path.append(PROJECT_A_PATH)

from project_a import ProjectAModel, preprocess_image


class ProjectABridge:
    """
    Adapter that translates Project B inputs to Project A's inference interface.

    - Dynamically injects `./project_a` onto `sys.path`.
    - Converts Project B base64 image payloads into numpy arrays for Project A.
    - Uses a singleton Project A model instance to avoid repeated allocations.
    """

    def __init__(self) -> None:
        self.model = self._get_model_instance()

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_model_instance() -> ProjectAModel:
        return ProjectAModel()

    def run_inference(self, base64_payload: str) -> Dict[str, Any]:
        processed = preprocess_image(base64_payload)
        result = self.model.predict(processed)
"""Adapter that bridges Project B requests to the Project A model."""

import base64
import os
import sys
from threading import Lock
from typing import Any, Dict, Optional

PROJECT_A_PATH = os.path.join(os.path.dirname(__file__), "project_a")
if PROJECT_A_PATH not in sys.path:
    sys.path.insert(0, PROJECT_A_PATH)

from project_a import inference as project_a_inference


class AIBridge:
    """Singleton bridge that adapts Project B payloads for Project A."""

    _instance: "AIBridge | None" = None

    def __new__(cls) -> "AIBridge":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = project_a_inference
        return cls._instance

    def execute(self, encoded_payload: str) -> Dict[str, Any]:
        """Convert Project B payload to Project A format and run inference."""
        image_array = self._decode_payload(encoded_payload)
        prediction = self._model.predict(image_array)
        return {"status": "success", "prediction": prediction}

    def _decode_payload(self, encoded_payload: str) -> np.ndarray:
        """Decode base64-encoded JSON data into a numpy array.

        Project B sends a base64 string containing JSON with ``width``,
        ``height``, and ``data`` fields. ``data`` is expected to be a flat
        list of grayscale pixel intensities.
        """
        decoded_bytes = base64.b64decode(encoded_payload.encode("utf-8"))
        payload = json.loads(decoded_bytes.decode("utf-8"))
        width = int(payload.get("width", 0))
        height = int(payload.get("height", 0))
        data: List[float] = payload.get("data", [])
        if width * height != len(data):
            raise ValueError("Invalid payload dimensions")
        array = np.array(data, dtype=float).reshape((height, width))
        return array
from project_a.model import ProjectAModel


class AIBridge:
    """Singleton adapter that translates Project B payloads into Project A inputs."""

    _instance: Optional["AIBridge"] = None
    _lock: Lock = Lock()

    def __init__(self) -> None:
        self.model = ProjectAModel()

    @classmethod
    def get_instance(cls) -> "AIBridge":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def predict_from_base64(self, payload: str) -> Dict[str, Any]:
        """Decode Project B base64 payloads and return Project A predictions."""

        decoded_text = self._decode_to_text(payload)
        result = self.model.predict(decoded_text)
        return {
            "status": "success",
            "data": result,
        }


__all__ = ["ProjectABridge", "ProjectBPayload"]
    @staticmethod
    def encode_prediction_image(prediction_map: Dict[str, Any]) -> str:
        buffer = BytesIO()
        image = Image.new("RGB", (200, 50), color=(0, 0, 0))
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    def _decode_to_text(self, payload: str) -> str:
        stripped_payload = payload.strip()
        try:
            decoded_bytes = base64.b64decode(stripped_payload, validate=True)
            return decoded_bytes.decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            # If payload is not valid base64, treat it as plain text coming from Project B.
            return stripped_payload
