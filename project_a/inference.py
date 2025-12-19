"""Public inference entrypoint for Project A."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .model import ProjectAModel, SimpleImageModel

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.json"

_model: SimpleImageModel | None = None


def load_model_config() -> Dict[str, Any]:
    """Load the model configuration relative to this module."""

    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _get_model() -> SimpleImageModel:
    global _model
    if _model is None:
        _model = SimpleImageModel()
    return _model


class ProjectAInference:
    """Lightweight proxy that exposes Project A predictions."""

    def __init__(self) -> None:
        self.model = _get_model()

    def predict(self, image_array: np.ndarray) -> Dict[str, Any]:
        return self.model.predict(image_array)


def predict(image_array: np.ndarray) -> Dict[str, Any]:
    """Proxy prediction through a lazily initialized model instance."""

    return ProjectAInference().predict(image_array)


__all__ = ["ProjectAInference", "ProjectAModel", "load_model_config", "predict"]
