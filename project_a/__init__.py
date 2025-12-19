"""Project A package exposing the demo AI model."""

from .inference import ProjectAModel, predict
from .model import SecurityInferenceModel

__all__ = ["ProjectAModel", "SecurityInferenceModel", "predict"]

from __future__ import annotations

from .model import ProjectAModel, predict

__all__ = ["ProjectAModel", "predict", "preprocess_image"]


def preprocess_image(*args, **kwargs):  # type: ignore[override]
    """Lazy wrapper to avoid importing image deps unless needed."""

    from .preprocess import preprocess_image as _preprocess_image

    return _preprocess_image(*args, **kwargs)

from project_a.inference import ProjectAModel, load_model_config, predict

__all__ = ["ProjectAModel", "load_model_config", "predict"]
from .inference import ProjectAModel, load_model_config
from .model import SecurityInferenceModel, predict

__all__ = [
    "ProjectAModel",
    "SecurityInferenceModel",
    "load_model_config",
    "predict",
]
