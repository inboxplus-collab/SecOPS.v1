"""Minimal Project A model with path-safe file access."""

import json
import os
from typing import Any, Dict

PROJECT_A_ROOT = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(PROJECT_A_ROOT, "config", "model_config.json")
WEIGHTS_PATH = os.path.join(PROJECT_A_ROOT, "weights", "dummy_weights.txt")


class ProjectAModel:
    """Demo model that simulates an AI inference pipeline."""

    def __init__(self) -> None:
        self.config = self._load_config()
        self.weights_checksum = self._load_weights_checksum()

    def _load_config(self) -> Dict[str, Any]:
        with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
            return json.load(config_file)

    def _load_weights_checksum(self) -> int:
        with open(WEIGHTS_PATH, "r", encoding="utf-8") as weights_file:
            # Calculate a tiny checksum to simulate validating the weights file.
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
