import base64
from bridge_logic import ProjectABridge


def main() -> None:
    try:
        bridge = ProjectABridge()
    except ModuleNotFoundError as exc:
        print("ModuleNotFoundError detected while loading bridge: ", exc)
        raise

    sample_text = "Suspicious login anomaly observed"
    encoded_payload = base64.b64encode(sample_text.encode("utf-8")).decode("utf-8")
    result = bridge.execute(encoded_payload)
    print("Sanity check result:")
    print(result)


if __name__ == "__main__":
    main()
