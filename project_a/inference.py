"""Public inference entrypoint for Project A."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from .model import ProjectAModel

PROJECT_ROOT = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "model_config.json")


def load_model_config() -> Dict[str, Any]:
    """Load the model configuration relative to this module."""

    if not os.path.exists(CONFIG_PATH):
        return {}

    with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
        return json.load(fp)


class ProjectAInference:
    """Lightweight proxy that exposes Project A predictions."""

    def __init__(self) -> None:
        self.model = ProjectAModel()

    def predict(self, text: str) -> Dict[str, Any]:
        return self.model.predict(text)


def predict(text: str) -> Dict[str, Any]:
    """Module-level convenience wrapper."""

    return ProjectAInference().predict(text)


__all__ = ["ProjectAInference", "ProjectAModel", "load_model_config", "predict"]
