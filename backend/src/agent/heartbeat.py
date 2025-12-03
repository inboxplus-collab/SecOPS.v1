import time
from typing import Any, Dict

import requests

from .agent import SERVER, load_node_id
from .collectors import collect_system_info


def send_heartbeat(node_id: str, payload: Dict[str, Any]) -> None:
    requests.post(
        f"{SERVER}/agent/heartbeat",
        json={"node_id": node_id, "system": payload},
        timeout=10,
    )


def run(interval_seconds: int = 60) -> None:
    node_id = load_node_id()
    while True:
        system_info = collect_system_info()
        send_heartbeat(node_id, system_info)
        time.sleep(interval_seconds)


if __name__ == "__main__":
    run()
