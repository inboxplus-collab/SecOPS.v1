from __future__ import annotations

import logging
import sys
from typing import Optional

from .config import settings


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger with a simple, production-friendly format.
    """
    log_level = level

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    # Example: make SQLAlchemy a bit quieter
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "secops")
