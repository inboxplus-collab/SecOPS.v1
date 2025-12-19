"""Public inference entrypoint for Project A."""

from typing import Any, Dict

import numpy as np

from .model import SimpleImageModel

_model: SimpleImageModel | None = None


def _get_model() -> SimpleImageModel:
    global _model
    if _model is None:
        _model = SimpleImageModel()
    return _model


def predict(image_array: np.ndarray) -> Dict[str, Any]:
    """Proxy prediction through a lazily initialized model instance."""
    model = _get_model()
    return model.predict(image_array)
