import base64
import io
from pprint import pprint

from PIL import Image

from bridge_logic import ProjectABridge


def generate_dummy_base64() -> str:
    image = Image.new("RGB", (224, 224), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def main() -> None:
    try:
        bridge = ProjectABridge()
        dummy_payload = generate_dummy_base64()
        prediction = bridge.run_inference(dummy_payload)
        pprint(prediction)
    except ModuleNotFoundError as exc:
        print(f"Module import failed: {exc}")
    except Exception as exc:  # noqa: BLE001 - surfaced during sanity check
        print(f"Unexpected error during sanity check: {exc}")
"""Quick sanity check for the Project A <-> Project B bridge."""

import base64
from pprint import pprint

from bridge_logic import AIBridge


def main() -> None:
    bridge = AIBridge.get_instance()

    sample_payload = base64.b64encode(b"This is a dummy Project B request").decode("utf-8")
    try:
        response = bridge.predict_from_base64(sample_payload)
    except ModuleNotFoundError as exc:  # pragma: no cover - explicit debugging aid
        print(f"Module import failed inside the bridge: {exc}")
        raise

    print("Bridge response:")
    pprint(response)


if __name__ == "__main__":
    main()
