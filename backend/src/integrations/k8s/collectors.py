from __future__ import annotations

from typing import Dict, List

from integrations.k8s.client import get_k8s_client


def collect_workloads() -> List[Dict]:
    """Return a lightweight representation of cluster workloads.

    The stub client returns empty lists when Kubernetes is unavailable, so this
    helper simply stitches those into the expected structure.
    """

    k8s = get_k8s_client()
    workloads: List[Dict] = []

    for dep in k8s.list_deployments().items:
        workloads.append({"metadata": {}, "spec": {}, "kind": "Deployment", "raw": dep})

    for st in k8s.list_statefulsets().items:
        workloads.append({"metadata": {}, "spec": {}, "kind": "StatefulSet", "raw": st})

    for ds in k8s.list_daemonsets().items:
        workloads.append({"metadata": {}, "spec": {}, "kind": "DaemonSet", "raw": ds})

    return workloads
