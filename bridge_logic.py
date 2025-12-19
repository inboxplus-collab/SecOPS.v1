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
                    cls._instance._model = ProjectAInference()
        return cls._instance

    def __init__(self) -> None:
        # Initialization handled in __new__ to enforce singleton behavior.
        pass

    def _extract_base64(self, payload: ProjectBPayload) -> str:
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
