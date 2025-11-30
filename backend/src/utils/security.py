from __future__ import annotations

import html
import re
from typing import Any, Mapping


SCRIPT_RE = re.compile(r"<\s*script", re.IGNORECASE)


def sanitize_html_fragment(value: str) -> str:
    """
    Very small helper to neutralize obvious script tags.

    This is NOT a full HTML sanitizer (use Bleach or similar for that if needed),
    but it saves you from the worst cases in descriptions.
    """
    if SCRIPT_RE.search(value):
        value = SCRIPT_RE.sub("&lt;script", value)
    return value


def redact_secrets(data: Mapping[str, Any]) -> dict[str, Any]:
    """
    Redact obvious secret-looking keys for logging/debugging.
    """
    redacted: dict[str, Any] = {}
    sensitive_keys = {"password", "secret", "token", "api_key", "authorization"}

    for key, value in data.items():
        if key.lower() in sensitive_keys:
            redacted[key] = "***redacted***"
        else:
            redacted[key] = value
    return redacted


def escape_for_log(value: str) -> str:
    """
    Escape newlines and control characters for safe log lines.
    """
    return html.escape(value, quote=True).replace("\n", "\\n")
