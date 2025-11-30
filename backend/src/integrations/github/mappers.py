from __future__ import annotations

from typing import Any, Dict

from core.checks.base import CheckResult


def check_result_to_issue_payload(result: CheckResult) -> Dict[str, Any]:
    return {
        "title": result.title,
        "description": result.description,
        "severity": result.severity,
        "status": "open",
        "source": "github",
        "metadata": result.metadata,
    }
