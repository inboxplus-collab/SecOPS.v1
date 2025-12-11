"""Dataset construction helpers for LLM fine-tuning."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence


def _normalize_prompt(content: str) -> str:
    prefix = (
        "You are SecOps AI, a security automation assistant. "
        "Summarize the issue, cite risky elements, and propose a precise mitigation."
    )
    return f"{prefix}\n\n<observation>\n{content.strip()}\n</observation>\n\n### Response:"


def build_dataset(files: Iterable[str], *, tags: Sequence[str] | None = None) -> List[dict]:
    """Convert raw text files into a JSONL-ready list of prompt/completion pairs.

    Each prompt is wrapped with an instruction header that nudges the model toward
    security-remediation style answers. Minimal metadata (tags) is embedded so the
    trainer can filter or bucket examples without depending on external schemas.
    """

    dataset: List[dict] = []
    for file_path in files:
        path = Path(file_path)
        with path.open("r", encoding="utf-8") as file:
            content = file.read()

        entry = {
            "prompt": _normalize_prompt(content),
            # Completion is intentionally short and factual; the trainer will learn to
            # expand responses using the instruction-heavy prompt above.
            "completion": "Provide a concise remediation plan grounded in the observation.",
            "meta": {
                "source": path.name,
                "tags": list(tags) if tags else [],
            },
        }
        dataset.append(entry)

    return dataset
