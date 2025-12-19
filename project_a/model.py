from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
CLASSIFIER_CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.json"
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "dummy_weights.txt"


class SecurityInferenceModel:
    """Lightweight heuristic model for text classification."""

    def __init__(self) -> None:
        with CLASSIFIER_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        self.labels: List[str] = config.get("labels", [])
        self.keywords: Dict[str, List[str]] = config.get("keywords", {})

    def predict(self, text: str) -> Dict[str, str]:
        normalized = text.lower()

        for label in ("critical", "suspicious"):
            for keyword in self.keywords.get(label, []):
                if keyword in normalized:
                    return {
                        "label": label,
                        "reason": f"Matched keyword '{keyword}'",
                    }

        return {"label": "benign", "reason": "No known threat indicators detected"}


def predict(text: str) -> Dict[str, str]:
    """Module-level helper mirroring the repository's exported API."""

    model = SecurityInferenceModel()
    return model.predict(text)


class ProjectAModel:
    """Demo model that simulates an AI inference pipeline."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or self._load_config()
        self.weights_checksum = self._load_weights_checksum()

    def _load_config(self) -> Dict[str, Any]:
        with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            return json.load(config_file)

    def _load_weights_checksum(self) -> int:
        with WEIGHTS_PATH.open("r", encoding="utf-8") as weights_file:
            return sum(ord(char) for char in weights_file.read())

    def predict(self, text: str) -> Dict[str, Any]:
        tokens = text.strip().split()
        prompt_template = self.config.get("prompt_template", "{text}")
        prompt = prompt_template.replace("{text}", text.strip())
        confidence = min(
            1.0,
            self.config.get("confidence_base", 0.5) + len(tokens) * 0.01,
        )
        return {
            "model": self.config.get("model_name", "project_a"),
            "provider": self.config.get("provider", "unknown"),
            "version": self.config.get("version", "unknown"),
            "weights_checksum": self.weights_checksum,
            "prompt": prompt,
            "tokens": tokens,
            "token_count": len(tokens),
            "confidence": round(confidence, 3),
        }
