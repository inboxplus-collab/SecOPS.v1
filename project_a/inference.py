from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .model import ProjectAModel

_PROJECT_ROOT = Path(__file__).resolve().parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "model_config.json"


def load_model_config() -> Dict[str, Any]:
    """Load the model configuration relative to this module."""
    with _CONFIG_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


class ProjectAModelWrapper:
    """Wrapper that keeps configuration loading centralized."""

    def __init__(self) -> None:
        self.config = load_model_config()
        self.model = ProjectAModel(self.config)

    def predict(self, text: str) -> Dict[str, Any]:
        return self.model.predict(text)


def predict(text: str) -> Dict[str, Any]:
    """Public inference entrypoint for Project A."""

    return ProjectAModelWrapper().predict(text)


__all__ = ["ProjectAModelWrapper", "ProjectAModel", "predict", "load_model_config"]
