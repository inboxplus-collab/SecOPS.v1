from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_PROJECT_ROOT = Path(__file__).resolve().parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "model_config.json"


def load_model_config() -> Dict[str, Any]:
    """Load the model configuration relative to this module."""
    with _CONFIG_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


class ProjectAModel:
    """Lightweight placeholder for Project A's inference engine."""

    def __init__(self) -> None:
        self.config = load_model_config()

    def predict(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        prompt = self.config.get("prompt_template", "{text}").replace("{text}", cleaned)
        return {
            "model": self.config.get("model_name", "unknown-model"),
            "provider": self.config.get("provider", "unknown-provider"),
            "prompt": prompt,
            "tokens_processed": max(1, len(cleaned.split())),
        }
"""Public inference entrypoint for Project A."""

from typing import Any, Dict

import numpy as np

from .model import SimpleImageModel

_model: SimpleImageModel | None = None


def _get_model() -> SimpleImageModel:
    global _model
    if _model is None:
        _model = SimpleImageModel()
    return _model


def predict(image_array: np.ndarray) -> Dict[str, Any]:
    """Proxy prediction through a lazily initialized model instance."""
    model = _get_model()
    return model.predict(image_array)
