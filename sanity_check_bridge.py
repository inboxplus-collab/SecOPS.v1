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
"""Sanity check for the Project A <-> Project B bridge."""

from __future__ import annotations

import base64
from pprint import pprint

from bridge_logic import AIBridge


def generate_dummy_base64() -> str:
def main() -> None:
    bridge = ProjectABridge()
    sample_payload = {"text": "This is a dummy Project B request"}
    encoded_payload = base64.b64encode(sample_payload["text"].encode("utf-8")).decode("utf-8")

    try:
        response = bridge.execute(encoded_payload)
    """Quick sanity check for the Project A <-> Project B bridge."""
    try:
        bridge = ProjectABridge()
    dummy_text = "This is a dummy Project B request"
    return base64.b64encode(dummy_text.encode("utf-8")).decode("utf-8")


def main() -> None:
    dummy_payload = generate_dummy_base64()

    try:
        bridge = AIBridge()
        response = bridge.execute(dummy_payload)
    except ModuleNotFoundError as exc:  # pragma: no cover - explicit debug aid
        print(f"Module import failed inside the bridge: {exc}")
        return
    except Exception as exc:  # noqa: BLE001 - surfaced during sanity check
        print(f"Unexpected error during sanity check: {exc}")
        return

    sample_payload = base64.b64encode(b"This is a dummy Project B request").decode("utf-8")
    response = bridge.execute(sample_payload)

    response = bridge.execute(encoded_payload)
    pprint(response)

    assert response.get("status") == "success"
    assert isinstance(response.get("data"), dict)
    print("Sanity check passed.")


if __name__ == "__main__":
    main()
