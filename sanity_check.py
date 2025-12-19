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
