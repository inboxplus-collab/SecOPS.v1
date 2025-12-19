"""Project A package exposing the demo AI model."""

from __future__ import annotations

from .model import ProjectAModel, predict

__all__ = ["ProjectAModel", "predict", "preprocess_image"]


def preprocess_image(*args, **kwargs):  # type: ignore[override]
    """Lazy wrapper to avoid importing image deps unless needed."""

    from .preprocess import preprocess_image as _preprocess_image

    return _preprocess_image(*args, **kwargs)
