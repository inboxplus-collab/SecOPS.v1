"""Project A package exposing the demo AI model."""

from .inference import ProjectAModel, predict
from .model import SecurityInferenceModel

__all__ = ["ProjectAModel", "SecurityInferenceModel", "predict"]
