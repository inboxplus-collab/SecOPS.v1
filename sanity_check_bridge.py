"""Sanity check for the Project A/B bridge layer."""

from __future__ import annotations

import base64
from pprint import pprint

try:
    from bridge_logic import ProjectABridge
except ModuleNotFoundError as exc:  # pragma: no cover - explicit failure surface
    print("ModuleNotFoundError encountered while importing bridge_logic:", exc)
    raise


def main() -> None:
    bridge = ProjectABridge()

    dummy_text = "Suspicious login anomaly observed"
    encoded_payload = base64.b64encode(dummy_text.encode("utf-8")).decode("utf-8")

    response = bridge.execute(encoded_payload)
    pprint(response)

    assert response.get("status") == "success"
    assert isinstance(response.get("data"), dict)
    print("Sanity check passed.")


if __name__ == "__main__":
    main()
