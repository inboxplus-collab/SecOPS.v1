"""Quick sanity check that Project B payloads reach Project A's inference."""

from __future__ import annotations

from pprint import pprint

try:
    from bridge_logic import ProjectABridge
except ModuleNotFoundError as exc:  # pragma: no cover - explicit failure surface
    print("ModuleNotFoundError detected while loading bridge_logic:", exc)
    raise


def main() -> None:
    bridge = ProjectABridge()

    # Project B often forwards decoded JSON payloads; mimic that format here.
    sample_payload = {"text": "Potential credential dumping detected"}
    response = bridge.execute(sample_payload)
"""Sanity check for the Project A/B bridge."""
"""Sanity check script for the Project A/B bridge."""

from __future__ import annotations

import base64
from pprint import pprint

from bridge_logic import ProjectABridge


def main() -> None:
    try:
        bridge = ProjectABridge()
    except ModuleNotFoundError as exc:  # pragma: no cover - debugging aid
        print(f"ModuleNotFoundError detected while loading bridge: {exc}")
    except ModuleNotFoundError as exc:  # pragma: no cover - explicit debugging aid
        print("ModuleNotFoundError detected while loading bridge: ", exc)
        raise

    sample_text = "Suspicious login anomaly observed"
    encoded_payload = base64.b64encode(sample_text.encode("utf-8")).decode("utf-8")
    result = bridge.execute(encoded_payload)
    pprint(result)
    response = bridge.execute(encoded_payload)
from bridge_logic import AIBridge


def main() -> None:
    bridge = AIBridge.get_instance()

    sample_text = "Suspicious login anomaly observed"
    encoded_payload = base64.b64encode(sample_text.encode("utf-8")).decode("utf-8")
    response = bridge.predict(encoded_payload)

    print("Sanity check result:")
    pprint(response)

    pprint(response)
    assert response.get("status") == "success"
    assert response.get("data", {}).get("token_count", 0) > 0
    print("Sanity check succeeded.")
    prediction = response.get("prediction", {})
    assert isinstance(prediction, dict)
    assert prediction.get("classification", {}).get("label")
    assert "token_count" in prediction
    print("Validation passed.")


if __name__ == "__main__":
    main()
