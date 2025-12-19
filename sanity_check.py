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

    pprint(response)
    assert response.get("status") == "success"
    assert response.get("data", {}).get("token_count", 0) > 0
    print("Sanity check succeeded.")


if __name__ == "__main__":
    main()
