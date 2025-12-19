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
import json
import os
from typing import Any, Dict, List

PROJECT_ROOT = os.path.dirname(__file__)
MODEL_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "model_config.json")
LABEL_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.json")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "weights", "dummy_weights.txt")


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

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
        with open(self.config_path, "r", encoding="utf-8") as config_file:
    def __init__(self) -> None:
        with CLASSIFIER_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config_path = CONFIG_DIR / "config.json"
        with config_path.open("r", encoding="utf-8") as config_file:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
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

        return {"label": "benign", "reason": "No known threat indicators detected"}


def predict(text: str) -> Dict[str, str]:
    """Module-level helper mirroring the repository's exported API."""

    model = SecurityInferenceModel()
    return model.predict(text)
    model = SecurityInferenceModel()
    return model.predict(text)


from __future__ import annotations

class ProjectATensorflowModel:
    """TensorFlow-based image classifier using on-disk config and weights."""

class ProjectAModel:
    """Demo AI model that classifies text inputs.

    The model keeps path resolution relative to this module so moving the
    project under a different directory does not break config or weight loads.
    """

    def __init__(self) -> None:
        self.model_config = _load_json(MODEL_CONFIG_PATH)
        label_config = _load_json(LABEL_CONFIG_PATH)
        self.labels: List[str] = label_config.get("labels", [])
        self.keyword_map: Dict[str, List[str]] = label_config.get("keywords", {})
        self.weights_checksum = self._load_weights_checksum()

    def __init__(self) -> None:
        self.base_dir = BASE_DIR
        self.config_path = CONFIG_DIR / "model_config.json"
        self.weights_path = self.base_dir / "artifacts" / "model.keras"
        self.labels = self._load_labels()
        self.model = self._load_model()

    def _load_labels(self) -> List[str]:
        if not self.config_path.exists():
            return ["label"]

        with self.config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("class_labels", ["label"])

    def _build_fallback_model(self, input_shape: tuple) -> Any:
        if tf is None:
            raise RuntimeError("TensorFlow is required for the fallback model")

        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Rescaling(1.0 / 255.0),
                tf.keras.layers.Conv2D(8, 3, activation="relu"),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(len(self.labels), activation="softmax"),
            ]
        )
        return model

    def _load_model(self) -> Any:
        input_shape = (224, 224, 3)
        if self.weights_path.exists():
            if tf is None:
                raise RuntimeError("TensorFlow is required to load saved models")
            return tf.keras.models.load_model(self.weights_path)

        return self._build_fallback_model(input_shape)

    def predict(self, processed_image: np.ndarray) -> Dict[str, Any]:
        if tf is None:
            raise RuntimeError("TensorFlow is required for TensorFlow model predictions")
        predictions = self.model.predict(processed_image)
        label_index = int(np.argmax(predictions, axis=1)[0])
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

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or self._load_config()
        self.weights_checksum = self._load_weights_checksum()

    def _load_config(self) -> Dict[str, Any]:
        with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            return json.load(config_file)

    def _load_weights_checksum(self) -> int:
        with WEIGHTS_PATH.open("r", encoding="utf-8") as weights_file:
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

    def _classify(self, normalized_text: str) -> str:
        for label in ("critical", "suspicious"):
            for keyword in self.keyword_map.get(label, []):
                if keyword in normalized_text:
                    return label
        return "benign"

    def predict(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        base_prediction = self._keywords_model.predict(cleaned)
        prompt = self.config.get("prompt_template", "{text}").replace("{text}", cleaned)
        confidence = min(1.0, self.config.get("confidence_base", 0.5) + len(cleaned.split()) * 0.01)
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
        normalized = text.strip().lower()
        tokens = normalized.split()
        label = self._classify(normalized)
        prompt_template = self.model_config.get("prompt_template", "{text}")
        prompt = prompt_template.replace("{text}", text.strip())
        confidence = min(1.0, self.model_config.get("confidence_base", 0.5) + len(tokens) * 0.01)

        return {
            "model": self.model_config.get("model_name", "project_a"),
            "version": self.model_config.get("version", "unknown"),
            "weights_checksum": self.weights_checksum,
            "label": label,
            "token_count": len(tokens),
            "prompt": prompt,
            "confidence": round(confidence, 3),
        tokens: List[str] = text.strip().split()
        confidence_base = float(self.config.get("confidence_base", 0.5))
        confidence = min(1.0, confidence_base + len(tokens) * 0.01)
        return {
            "model": self.config.get("model_name", "project_a"),
            "provider": self.config.get("provider", "unknown"),
            "prompt": prompt,
            "tokens_processed": max(1, len(cleaned.split())),
            "confidence": round(confidence, 3),
            **base_prediction,
        }
        return {"label": "benign", "reason": "No known threat indicators detected"}


def predict(text: str) -> Dict[str, Any]:
    model = ProjectAModel()
    return model.predict(text)


__all__ = ["ProjectAModel", "predict"]
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


__all__ = [
    "ProjectAModel",
    "SecurityInferenceModel",
    "SimpleImageModel",
]
