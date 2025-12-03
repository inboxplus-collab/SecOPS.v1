import os
import sys
import time
from typing import Any, Dict, List

import requests

try:  # pragma: no cover - support both package and standalone execution
    from .collectors import collect_logs, collect_system_info
    from .auto_fix import auto_patch
except ImportError:  # pragma: no cover
    from collectors import collect_logs, collect_system_info
    from auto_fix import auto_patch

SERVER = os.getenv("SECOPSAI_SERVER", "https://api.secops.ai")
CONFIG_PATH = os.getenv(
    "SECOPSAI_CONFIG",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent.conf"),
)


def _load_config() -> Dict[str, str]:
    if not os.path.exists(CONFIG_PATH):
        return {}

    data: Dict[str, str] = {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as cfg:
        for line in cfg:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                data[key] = value
    return data


def _persist_config(entries: Dict[str, str]) -> None:
    existing = _load_config()
    existing.update(entries)

    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as cfg:
        for key, value in existing.items():
            cfg.write(f"{key}={value}\n")


def register() -> str:
    resp = requests.post(
        f"{SERVER}/agent/register",
        json={"hostname": os.uname().nodename},
        timeout=15,
    )
    resp.raise_for_status()

    node_id = resp.json().get("node_id")
    if not node_id:
        raise RuntimeError("Registration failed: missing node_id")

    _persist_config({"NODE_ID": node_id})
    return node_id


def load_node_id() -> str:
    env_node = os.getenv("SECOPSAI_NODE_ID")
    if env_node:
        return env_node

    config = _load_config()
    if node_id := config.get("NODE_ID"):
        return node_id

    return register()


def _post_heartbeat(node_id: str, logs: List[str], info: Dict[str, Any]) -> None:
    requests.post(
        f"{SERVER}/agent/heartbeat",
        json={"node_id": node_id, "logs": logs, "system": info},
        timeout=15,
    )


def _fetch_commands(node_id: str) -> List[Dict[str, Any]]:
    resp = requests.get(f"{SERVER}/agent/commands/{node_id}", timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    return []


def run() -> None:
    node_id = load_node_id()

    while True:
        try:
            logs = collect_logs()
            info = collect_system_info()

            _post_heartbeat(node_id, logs, info)

            for cmd in _fetch_commands(node_id):
                if cmd.get("type") == "AUTO_FIX":
                    auto_patch(cmd.get("file_path"), cmd.get("patch"))
        except Exception as exc:  # pragma: no cover - keep agent resilient
            print(f"[SecOpsAI Agent] Error: {exc}")
        time.sleep(5)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "register":
        print(register())
    else:
        run()
