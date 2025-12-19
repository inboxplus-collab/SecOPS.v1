"""
Project A package exposing model loading and inference utilities.
"""

from .model import ProjectAModel, load_model
from .preprocess import preprocess_image

__all__ = ["ProjectAModel", "load_model", "preprocess_image"]
"""Project A package exposing the demo AI model."""

from .model import ProjectAModel, predict

__all__ = ["ProjectAModel", "predict"]
