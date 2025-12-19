"""Project A: Local model stub used for bridge integration tests."""

from project_a.inference import ProjectAModel, load_model_config  # noqa: F401
"""Project A minimal inference package."""
"""
Project A package exposing model loading and inference utilities.
"""

from .model import ProjectAModel, load_model
from .preprocess import preprocess_image

__all__ = ["ProjectAModel", "load_model", "preprocess_image"]
"""Project A package exposing the demo AI model."""

from .model import ProjectAModel, predict

__all__ = ["ProjectAModel", "predict"]
