from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .model import SimpleImageModel

_PROJECT_ROOT = Path(__file__).resolve().parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "model_config.json"
_model: SimpleImageModel | None = None


def load_model_config() -> Dict[str, Any]:
    """Load the model configuration relative to this module."""
    with _CONFIG_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _get_model() -> SimpleImageModel:
    global _model
    if _model is None:
        _model = SimpleImageModel()
    return _model


def predict(image_array: np.ndarray) -> Dict[str, Any]:
    """Proxy prediction through a lazily initialized model instance."""
    model = _get_model()
    return model.predict(image_array)


class ProjectAModel:
    """Alias wrapper for compatibility with upstream imports."""

    def __init__(self) -> None:
        self._model = _get_model()

    def predict(self, image_array: np.ndarray) -> Dict[str, Any]:
        return self._model.predict(image_array)

