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
