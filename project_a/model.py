"""Project A model implementation with path-safe resource loading."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"


class SecurityInferenceModel:
    """Lightweight heuristic model standing in for a heavier AI model.

    The model configuration is loaded from a JSON file relative to this module so
    relocating the project under a new directory continues to work.
    """

    def __init__(self) -> None:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        self.labels: List[str] = config.get("labels", [])
        self.keywords: Dict[str, List[str]] = config.get("keywords", {})

    def predict(self, text: str) -> Dict[str, str]:
        """Return a simple classification and reason string for the input text."""
        normalized = text.lower()

from __future__ import annotations

import json
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
            "model": self.config.get("model_name", "project_a"),
            "version": self.config.get("version", "unknown"),
            "weights_checksum": self.weights_checksum,
            "tokens": tokens,
            "token_count": len(tokens),
            "confidence": round(confidence, 3),
        }
        return {"label": "benign", "reason": "No known threat indicators detected"}


def predict(text: str) -> Dict[str, str]:
    """Module-level helper mirroring the repository's exported API."""

    model = SecurityInferenceModel()
    return model.predict(text)


    return ProjectAModel().predict(text)


__all__ = ["ProjectAModel", "predict"]
__all__ = ["SecurityInferenceModel", "predict"]
