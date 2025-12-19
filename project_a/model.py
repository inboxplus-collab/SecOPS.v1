from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"


class SecurityInferenceModel:
    """Lightweight heuristic model standing in for a heavier AI model.

    The model configuration is loaded from a JSON file relative to this module so
    relocating the project under a new directory continues to work.
    """

    def __init__(self) -> None:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        self.labels: List[str] = config.get("labels", [])
        self.keywords: Dict[str, List[str]] = config.get("keywords", {})

    def predict(self, text: str) -> Dict[str, str]:
        """Return a simple classification and reason string for the input text."""
        normalized = text.lower()

        for label in ("critical", "suspicious"):
            for keyword in self.keywords.get(label, []):
                if keyword in normalized:
                    return {
                        "label": label,
                        "reason": f"Matched keyword '{keyword}'",
                    }

        return {"label": "benign", "reason": "No known threat indicators detected"}


def predict(text: str) -> Dict[str, str]:
    """Module-level helper mirroring the repository's exported API."""

    model = SecurityInferenceModel()
    return model.predict(text)


__all__ = ["SecurityInferenceModel", "predict"]
