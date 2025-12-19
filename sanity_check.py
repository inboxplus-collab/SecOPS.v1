"""Sanity check script for the Project A/B bridge."""

from __future__ import annotations

import base64
from pprint import pprint

from bridge_logic import AIBridge


def main() -> None:
    bridge = AIBridge.get_instance()

    sample_text = "Suspicious login anomaly observed"
    encoded_payload = base64.b64encode(sample_text.encode("utf-8")).decode("utf-8")
    response = bridge.predict(encoded_payload)

    print("Sanity check result:")
    pprint(response)

    assert response.get("status") == "success"
    prediction = response.get("prediction", {})
    assert isinstance(prediction, dict)
    assert "token_count" in prediction
    print("Validation passed.")


if __name__ == "__main__":
    main()
