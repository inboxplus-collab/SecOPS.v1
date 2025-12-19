from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import tensorflow as tf
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tf = None

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
WEIGHTS_DIR = BASE_DIR / "weights"


class SecurityInferenceModel:
    """Lightweight heuristic model standing in for a heavier AI model."""

    def __init__(self) -> None:
        config_path = CONFIG_DIR / "config.json"
        with config_path.open("r", encoding="utf-8") as config_file:
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


class ProjectATensorflowModel:
    """TensorFlow-based image classifier using on-disk config and weights."""

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
            "label": self.labels[label_index] if label_index < len(self.labels) else str(label_index),
            "confidence": float(predictions[0][label_index]),
        }


class SimpleImageModel:
    """Toy model that scores an array and returns a label."""

    def __init__(self) -> None:
        data_dir = DATA_DIR
        with (data_dir / "config.json").open("r", encoding="utf-8") as config_file:
            self.config = json.load(config_file)
        with (data_dir / "weights.json").open("r", encoding="utf-8") as weights_file:
            self.weights = json.load(weights_file)

    def predict(self, image_array: np.ndarray) -> Dict[str, Any]:
        normalized = (image_array - image_array.min()) / max(image_array.ptp(), 1e-8)
        score = float(normalized.mean()) * self.weights["mean_multiplier"]
        label = "high-signal" if score > self.config["threshold"] else "low-signal"
        return {"label": label, "score": round(score, 4)}


class ProjectAModel:
    """Demo model that simulates an AI inference pipeline."""

    def __init__(self) -> None:
        self.config_path = CONFIG_DIR / "model_config.json"
        self.weights_path = WEIGHTS_DIR / "dummy_weights.txt"
        self.config = self._load_config()
        self.weights_checksum = self._load_weights_checksum()

    def _load_config(self) -> Dict[str, Any]:
        with self.config_path.open("r", encoding="utf-8") as config_file:
            return json.load(config_file)

    def _load_weights_checksum(self) -> int:
        with self.weights_path.open("r", encoding="utf-8") as weights_file:
            return sum(ord(char) for char in weights_file.read())

    def predict(self, text: str) -> Dict[str, Any]:
        tokens = text.strip().split()
        confidence = min(1.0, self.config.get("confidence_base", 0.5) + len(tokens) * 0.01)
        return {
            "model": self.config.get("model_name", "project_a"),
            "version": self.config.get("version", "unknown"),
            "weights_checksum": self.weights_checksum,
            "tokens": tokens,
            "token_count": len(tokens),
            "confidence": round(confidence, 3),
        }


def predict(text: str) -> Dict[str, Any]:
    model = ProjectAModel()
    return model.predict(text)

