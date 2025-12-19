"""Project A package exposing the demo AI model and preprocessing utilities."""
"""Project A package exposing the demo AI model."""

from __future__ import annotations

from .inference import ProjectAModelWrapper, load_model_config, predict as inference_predict
from .model import ProjectAModel, SecurityInferenceModel, predict

try:  # Optional dependency: only present when image preprocessing is needed
    from .preprocess import preprocess_image
except ModuleNotFoundError:  # pragma: no cover - optional at runtime
    preprocess_image = None

__all__ = [
    "ProjectAModel",
    "ProjectAModelWrapper",
    "SecurityInferenceModel",
    "predict",
    "inference_predict",
    "load_model_config",
    "preprocess_image",

from .inference import ProjectAModel, load_model_config

__all__ = ["ProjectAModel", "load_model_config"]
"""Project A package exposing the demo AI model."""

from .inference import ProjectAModel, load_model_config, predict
from .model import SecurityInferenceModel

__all__ = ["ProjectAModel", "load_model_config", "predict", "SecurityInferenceModel"]
"""Project A package exposing model loading and inference utilities."""

from .inference import ProjectAModel, predict  # noqa: F401
from .preprocess import preprocess_image  # noqa: F401

__all__ = ["ProjectAModel", "predict", "preprocess_image"]

"""Project A package exposing the demo AI model."""

from .inference import ProjectAModel, predict
from .model import SecurityInferenceModel

__all__ = ["ProjectAModel", "SecurityInferenceModel", "predict"]

from __future__ import annotations

from .inference import ProjectAInference, load_model_config, predict
from .model import ProjectAModel, SecurityInferenceModel
from .preprocess import preprocess_image

__all__ = [
    "ProjectAInference",
    "ProjectAModel",
    "SecurityInferenceModel",
    "load_model_config",
    "predict",
    "preprocess_image",
]
