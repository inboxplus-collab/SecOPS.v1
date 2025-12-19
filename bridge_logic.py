"""Adapter that bridges Project B requests to the Project A model."""

import base64
import os
import sys
from threading import Lock
from typing import Any, Dict, Optional

PROJECT_A_PATH = os.path.join(os.path.dirname(__file__), "project_a")
if PROJECT_A_PATH not in sys.path:
    sys.path.insert(0, PROJECT_A_PATH)

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

    def _decode_to_text(self, payload: str) -> str:
        stripped_payload = payload.strip()
        try:
            decoded_bytes = base64.b64decode(stripped_payload, validate=True)
            return decoded_bytes.decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            # If payload is not valid base64, treat it as plain text coming from Project B.
            return stripped_payload
