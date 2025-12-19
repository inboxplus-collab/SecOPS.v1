"""Project A model definitions with path-safe file resolution."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

PROJECT_ROOT = os.path.dirname(__file__)
KEYWORD_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.json")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "weights", "dummy_weights.txt")


class SecurityInferenceModel:
    """Keyword-driven classifier used to emulate a heavier AI model."""

    def __init__(self, config_path: str | None = None, weights_path: str | None = None) -> None:
        self.config_path = config_path or KEYWORD_CONFIG_PATH
        self.weights_path = weights_path or WEIGHTS_PATH
        self.labels, self.keywords = self._load_keywords()
        self.weights_checksum = self._load_weights_checksum()

    def _load_keywords(self) -> tuple[List[str], Dict[str, List[str]]]:
        with open(self.config_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        labels: List[str] = config.get("labels", [])
        keywords: Dict[str, List[str]] = config.get("keywords", {})
        return labels, keywords

    def _load_weights_checksum(self) -> int:
        with open(self.weights_path, "r", encoding="utf-8") as weights_file:
            return sum(ord(char) for char in weights_file.read())

    def predict(self, text: str) -> Dict[str, Any]:
        """Return a label and metadata for the supplied text."""

        normalized = text.lower()
        for label in ("critical", "suspicious"):
            for keyword in self.keywords.get(label, []):
                if keyword in normalized:
                    return {
                        "label": label,
                        "reason": f"Matched keyword '{keyword}'",
                        "weights_checksum": self.weights_checksum,
                    }

        return {
            "label": "benign",
            "reason": "No known threat indicators detected",
            "weights_checksum": self.weights_checksum,
        }


def predict(text: str) -> Dict[str, Any]:
    """Convenience wrapper mirroring the repository's exported API."""

    model = SecurityInferenceModel()
    return model.predict(text)
