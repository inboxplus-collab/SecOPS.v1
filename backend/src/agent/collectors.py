import os
import platform
import socket
import subprocess
from typing import Any, Dict, List


def collect_logs(limit: int = 100) -> List[str]:
    """Collect the most recent system logs where available."""
    logs: List[str] = []
    syslog_paths = [
        "/var/log/syslog",
        "/var/log/messages",
        "/var/log/system.log",
    ]

    for path in syslog_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as log:
                    lines = log.readlines()[-limit:]
                    logs.extend([line.strip() for line in lines])
                    break
            except Exception:
                continue

    if not logs:
        logs.append("No system logs available or insufficient permissions.")

    return logs


def collect_system_info() -> Dict[str, Any]:
    """Gather lightweight system metadata for heartbeat payloads."""
    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "cpu": platform.processor(),
    }

    try:
        info["uptime"] = _get_uptime()
    except Exception:
        info["uptime"] = None

    try:
        info["load_avg"] = os.getloadavg()
    except Exception:
        info["load_avg"] = None

    return info


def _get_uptime() -> float:
    if os.name == "posix" and os.path.exists("/proc/uptime"):
        with open("/proc/uptime", "r", encoding="utf-8") as uptime_file:
            return float(uptime_file.readline().split()[0])

    result = subprocess.run(
        ["uptime"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip()
