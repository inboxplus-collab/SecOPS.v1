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


if __name__ == "__main__":
    main()
