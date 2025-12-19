"""Sanity check for the Project A <-> Project B bridge."""
from __future__ import annotations
"""Quick sanity check for the Project A <-> Project B bridge."""

from __future__ import annotations

import base64
from pprint import pprint

from bridge_logic import ProjectABridge


def main() -> None:
    dummy_text = "This is a dummy Project B request"
    encoded_payload = base64.b64encode(dummy_text.encode("utf-8")).decode("utf-8")

    try:
        bridge = ProjectABridge()
        dummy_payload = generate_dummy_base64()
        response = bridge.execute(dummy_payload)
        print("Bridge response:")
        pprint(response)
    except ModuleNotFoundError as exc:  # pragma: no cover - explicit debug aid
        print(f"Module import failed: {exc}")
    except Exception as exc:  # pragma: no cover - surfaced during sanity check
        print(f"Unexpected error during sanity check: {exc}")
        response = bridge.execute(encoded_payload)
    except ModuleNotFoundError as exc:  # pragma: no cover - explicit debugging aid
        print(f"Module import failed inside the bridge: {exc}")
        return
    except Exception as exc:  # noqa: BLE001 - surfaced during sanity check
        print(f"Unexpected error during sanity check: {exc}")
        return
    bridge = ProjectABridge()

    # Simulate Project B sending a base64-encoded text payload
    sample_payload = base64.b64encode(b"This is a dummy Project B request").decode("utf-8")

    try:
        bridge = AIBridge.get_instance()
        response = bridge.predict(sample_payload)
        response = bridge.execute(sample_payload)
    except ModuleNotFoundError as exc:  # pragma: no cover - explicit debugging aid
        print(f"Module import failed inside the bridge: {exc}")
        raise
    except Exception as exc:  # pragma: no cover - surfaced during sanity check
        print(f"Unexpected error during sanity check: {exc}")
        raise

    print("Bridge response:")
    pprint(response)


if __name__ == "__main__":
    main()

