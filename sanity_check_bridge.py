"""Sanity check for the Project A <-> Project B bridge."""
from __future__ import annotations

import base64
from pprint import pprint

from bridge_logic import ProjectABridge


def main() -> None:
    dummy_text = "This is a dummy Project B request"
    encoded_payload = base64.b64encode(dummy_text.encode("utf-8")).decode("utf-8")

    try:
        bridge = ProjectABridge()
        response = bridge.execute(encoded_payload)
    except ModuleNotFoundError as exc:  # pragma: no cover - explicit debugging aid
        print(f"Module import failed inside the bridge: {exc}")
        return
    except Exception as exc:  # noqa: BLE001 - surfaced during sanity check
        print(f"Unexpected error during sanity check: {exc}")
        return

    print("Bridge response:")
    pprint(response)


if __name__ == "__main__":
    main()
