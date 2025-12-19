"""Input preprocessing helpers for Project A."""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Tuple

import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "config", "model_config.json")


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        return {}

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_image(image_array: np.ndarray, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> np.ndarray:
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    normalized = (image_array / 255.0 - mean_arr) / std_arr
    return normalized.astype(np.float32)


def decode_base64_image(b64_string: str) -> Image.Image:
    decoded = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(decoded)).convert("RGB")


def preprocess_image(b64_string: str) -> np.ndarray:
    config = load_config()
    input_size = config.get("input_size", [224, 224, 3])
    target_size = tuple(input_size[:2])
    normalization_mean = tuple(config.get("normalization_mean", (0.0, 0.0, 0.0)))
    normalization_std = tuple(config.get("normalization_std", (1.0, 1.0, 1.0)))

    image = decode_base64_image(b64_string)
    resized = image.resize(target_size)
    array = np.array(resized, dtype=np.float32)
    normalized = _normalize_image(array, normalization_mean, normalization_std)
    normalized = normalized.reshape((1, *normalized.shape))
    return normalized


__all__ = [
    "decode_base64_image",
    "load_config",
    "preprocess_image",
]
