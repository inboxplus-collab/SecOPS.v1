import base64
from bridge_logic import ProjectABridge


def main() -> None:
    try:
        bridge = ProjectABridge()
    except ModuleNotFoundError as exc:
        print("ModuleNotFoundError detected while loading bridge: ", exc)
        raise

    sample_text = "Suspicious login anomaly observed"
    encoded_payload = base64.b64encode(sample_text.encode("utf-8")).decode("utf-8")
    result = bridge.execute(encoded_payload)
    print("Sanity check result:")
    print(result)

from __future__ import annotations

import base64
import json
from pprint import pprint

try:
    from bridge_logic import ProjectABridge
except ModuleNotFoundError as exc:
    print(f"ModuleNotFoundError encountered: {exc}")
    raise


def main() -> None:
    bridge = ProjectABridge()
    dummy_payload = base64.b64encode(b"dummy security event").decode("utf-8")
    response = bridge.execute(dummy_payload)

    print("Sanity check response:")
    pprint(response)

    assert response.get("status") == "success"
    data = response.get("data", {})
    assert isinstance(data, dict)
    assert "model" in data
    assert "prompt" in data
    print("Validation passed.")
"""Sanity check for the Project A/B bridge."""

import base64
import json
import sys

from bridge_logic import AIBridge


def main() -> None:
    dummy_payload = {
        "width": 4,
        "height": 4,
        "data": [
            0, 32, 64, 96,
            24, 56, 88, 120,
            48, 80, 112, 144,
            72, 104, 136, 168,
        ],
    }
    encoded = base64.b64encode(json.dumps(dummy_payload).encode("utf-8")).decode("utf-8")
    try:
        bridge = AIBridge()
        result = bridge.execute(encoded)
        print("Bridge output:", result)
    except ModuleNotFoundError as exc:
        print("ModuleNotFoundError detected:", exc, file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
