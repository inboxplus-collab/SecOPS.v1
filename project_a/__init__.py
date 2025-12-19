"""Project A package exposing the demo AI model."""

from .inference import ProjectAModel, load_model_config
from .model import SecurityInferenceModel, predict

__all__ = [
    "ProjectAModel",
    "SecurityInferenceModel",
    "load_model_config",
    "predict",
]
