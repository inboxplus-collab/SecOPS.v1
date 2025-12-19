"""Project A model definitions with path-safe file resolution."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

PROJECT_ROOT = os.path.dirname(__file__)
KEYWORD_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.json")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "weights", "dummy_weights.txt")
"""Project A demo model with path-safe configuration loading."""
"""Project A model implementation with path-safe resource loading."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"


class SecurityInferenceModel:
    """Keyword-driven classifier used to emulate a heavier AI model."""

    def __init__(self, config_path: str | None = None, weights_path: str | None = None) -> None:
        self.config_path = config_path or KEYWORD_CONFIG_PATH
        self.weights_path = weights_path or WEIGHTS_PATH
        self.labels, self.keywords = self._load_keywords()
        self.weights_checksum = self._load_weights_checksum()

    def _load_keywords(self) -> tuple[List[str], Dict[str, List[str]]]:
        with open(self.config_path, "r", encoding="utf-8") as config_file:
    def __init__(self) -> None:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
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
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.json"
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "dummy_weights.txt"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _calculate_checksum(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(ord(char) for char in handle.read())


class ProjectAModel:
    """Demo model that simulates an AI inference pipeline.

    The class loads lightweight configuration and a dummy weight file using
    paths relative to the module location so relocating ``project_a`` keeps
    file lookups stable.
    """

    def __init__(self) -> None:
        self.config = _load_json(CONFIG_PATH)
        self.weights_checksum = _calculate_checksum(WEIGHTS_PATH)

    def predict(self, text: str) -> Dict[str, Any]:
        tokens: List[str] = text.strip().split()
        confidence_base = float(self.config.get("confidence_base", 0.5))
        confidence = min(1.0, confidence_base + len(tokens) * 0.01)
        return {
            "label": "benign",
            "reason": "No known threat indicators detected",
            "weights_checksum": self.weights_checksum,
        }
        return {"label": "benign", "reason": "No known threat indicators detected"}


def predict(text: str) -> Dict[str, Any]:
    """Convenience wrapper mirroring the repository's exported API."""
def predict(text: str) -> Dict[str, str]:
    """Module-level helper mirroring the repository's exported API."""

    model = SecurityInferenceModel()
    return model.predict(text)


def load_model() -> ProjectAModel:
    """Factory helper for compatibility with older imports."""

    return ProjectAModel()


__all__ = ["ProjectAModel", "predict", "load_model"]
    return ProjectAModel().predict(text)


__all__ = ["ProjectAModel", "predict"]
__all__ = ["SecurityInferenceModel", "predict"]
