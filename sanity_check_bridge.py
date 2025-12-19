from __future__ import annotations

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
        response = bridge.execute(dummy_payload)
        print("Bridge response:")
        pprint(response)
    except ModuleNotFoundError as exc:  # pragma: no cover - explicit debug aid
        print(f"Module import failed: {exc}")
    except Exception as exc:  # pragma: no cover - surfaced during sanity check
        print(f"Unexpected error during sanity check: {exc}")


if __name__ == "__main__":
    main()

