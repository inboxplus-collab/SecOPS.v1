"""Public inference entrypoint for Project A."""
from __future__ import annotations

import json
import os
from typing import Any, Dict

from .model import SecurityInferenceModel

PROJECT_ROOT = os.path.dirname(__file__)
MODEL_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "model_config.json")


def load_model_config() -> Dict[str, Any]:
    """Load the model configuration relative to this module."""

    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as fp:
        return json.load(fp)


class ProjectAModel:
    """Lightweight placeholder for Project A's inference engine."""

    def __init__(self) -> None:
        self.config = load_model_config()
        self.model = SecurityInferenceModel()

    def predict(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        base_prediction = self.model.predict(cleaned)
        prompt = self.config.get("prompt_template", "{text}").replace("{text}", cleaned)
        confidence = min(1.0, self.config.get("confidence_base", 0.5) + len(cleaned.split()) * 0.01)
        return {
            "model": self.config.get("model_name", "unknown-model"),
            "provider": self.config.get("provider", "unknown-provider"),
            "prompt": prompt,
            "tokens_processed": max(1, len(cleaned.split())),
            "confidence": round(confidence, 3),
            **base_prediction,
        }


def predict(text: str) -> Dict[str, Any]:
    """Proxy prediction through a Project A model instance."""

    model = ProjectAModel()
    return model.predict(text)
