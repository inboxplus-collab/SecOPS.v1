import json
import os
from typing import Dict, List


class SecurityInferenceModel:
    """Lightweight heuristic model standing in for a heavier AI model.

    The model configuration is loaded from a JSON file relative to this module so
    relocating the project under a new directory continues to work.
    """

    def __init__(self) -> None:
        config_path = os.path.join(os.path.dirname(__file__), "config", "config.json")
        with open(config_path, "r", encoding="utf-8") as config_file:
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
