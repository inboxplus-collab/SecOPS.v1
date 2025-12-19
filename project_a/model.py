"""Project A demo model with path-safe configuration loading."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

PROJECT_A_ROOT = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(PROJECT_A_ROOT, "config", "model_config.json")
WEIGHTS_PATH = os.path.join(PROJECT_A_ROOT, "weights", "dummy_weights.txt")


def _load_json(path: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    if not os.path.exists(path):
        return fallback

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class ProjectAModel:
    """Demo model that simulates an AI inference pipeline."""

    def __init__(self) -> None:
        self.config = _load_json(
            CONFIG_PATH,
            {"model_name": "project_a", "version": "unknown", "confidence_base": 0.5},
        )
        self.weights_checksum = self._load_weights_checksum()

    def _load_weights_checksum(self) -> int:
        if not os.path.exists(WEIGHTS_PATH):
            return 0

        with open(WEIGHTS_PATH, "r", encoding="utf-8") as weights_file:
            return sum(ord(char) for char in weights_file.read())

    def predict(self, text: str) -> Dict[str, Any]:
        tokens = text.strip().split()
        confidence = min(1.0, self.config.get("confidence_base", 0.5) + len(tokens) * 0.01)
        return {
            "model": self.config.get("model_name", "project_a"),
            "version": self.config.get("version", "unknown"),
            "weights_checksum": self.weights_checksum,
            "tokens": tokens,
            "token_count": len(tokens),
            "confidence": round(confidence, 3),
        }


def predict(text: str) -> Dict[str, Any]:
    """Convenience wrapper matching the Project A public API."""

    model = ProjectAModel()
    return model.predict(text)


def load_model() -> ProjectAModel:
    """Factory helper for compatibility with older imports."""

    return ProjectAModel()


__all__ = ["ProjectAModel", "predict", "load_model"]
