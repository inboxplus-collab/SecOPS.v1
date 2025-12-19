"""Quick sanity check for the Project A <-> Project B bridge."""

from __future__ import annotations

import base64
from pprint import pprint

from bridge_logic import ProjectABridge


def main() -> None:
    bridge = ProjectABridge()

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
