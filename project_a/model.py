from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Resolve paths relative to this file to remain robust after nesting under ./project_a
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

KEYWORD_CONFIG_PATH = CONFIG_DIR / "config.json"
MODEL_CONFIG_PATH = CONFIG_DIR / "model_config.json"
WEIGHTS_PATH = WEIGHTS_DIR / "dummy_weights.txt"


class SecurityInferenceModel:
    """Keyword-driven classifier used to emulate a heavier AI model."""

    def __init__(self, config_path: Path | None = None, weights_path: Path | None = None) -> None:
        self.config_path = config_path or KEYWORD_CONFIG_PATH
        self.weights_path = weights_path or WEIGHTS_PATH
        self.labels, self.keywords = self._load_keywords()
        self.weights_checksum = self._load_weights_checksum()

    def _load_keywords(self) -> tuple[List[str], Dict[str, List[str]]]:
        with self.config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        labels: List[str] = config.get("labels", [])
        keywords: Dict[str, List[str]] = config.get("keywords", {})
        return labels, keywords

    def _load_weights_checksum(self) -> int:
        if not self.weights_path.exists():
            return 0
        with self.weights_path.open("r", encoding="utf-8") as weights_file:
            return sum(ord(char) for char in weights_file.read())

    def predict(self, text: str) -> Dict[str, Any]:
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
            "reason": "No keywords matched",
            "weights_checksum": self.weights_checksum,
        }


class SimpleImageModel:
    """Toy model that scores an array and returns a label."""

    def __init__(self) -> None:
        with (DATA_DIR / "config.json").open("r", encoding="utf-8") as config_file:
            self.config = json.load(config_file)
        with (DATA_DIR / "weights.json").open("r", encoding="utf-8") as weights_file:
            self.weights = json.load(weights_file)

    def predict(self, image_array: np.ndarray) -> Dict[str, Any]:
        normalized = (image_array - image_array.min()) / max(image_array.ptp(), 1e-8)
        score = float(normalized.mean()) * self.weights["mean_multiplier"]
        label = "high-signal" if score > self.config["threshold"] else "low-signal"
        return {"label": label, "score": round(score, 4)}


class ProjectAModel:
    """Demo model that simulates an AI inference pipeline."""

    def __init__(self) -> None:
        self.config_path = MODEL_CONFIG_PATH
        self.weights_path = WEIGHTS_PATH
        self.config = self._load_config()
        self.weights_checksum = self._load_weights_checksum()
        self._keywords_model = SecurityInferenceModel()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {"model_name": "project_a", "version": "unknown", "confidence_base": 0.5}
        with self.config_path.open("r", encoding="utf-8") as config_file:
            return json.load(config_file)

    def _load_weights_checksum(self) -> int:
        if not self.weights_path.exists():
            return 0
        with self.weights_path.open("r", encoding="utf-8") as weights_file:
            return sum(ord(char) for char in weights_file.read())

    def predict(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        base_prediction = self._keywords_model.predict(cleaned)
        prompt = self.config.get("prompt_template", "{text}").replace("{text}", cleaned)
        confidence = min(1.0, self.config.get("confidence_base", 0.5) + len(cleaned.split()) * 0.01)
        return {
            "model": self.config.get("model_name", "project_a"),
            "provider": self.config.get("provider", "unknown"),
            "prompt": prompt,
            "tokens_processed": max(1, len(cleaned.split())),
            "confidence": round(confidence, 3),
            **base_prediction,
        }


__all__ = [
    "ProjectAModel",
    "SecurityInferenceModel",
    "SimpleImageModel",
]
