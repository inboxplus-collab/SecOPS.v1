"""
Collectors that turn raw Kubernetes objects into normalized dicts
that the checks & healer can reason about.
"""

from __future__ import annotations

from typing import Dict, List

from backend.src.integrations.k8s.client import get_k8s_client


def _workload_metadata(obj) -> Dict:
    meta = obj.metadata
    return {
        "name": meta.name,
        "namespace": meta.namespace or "default",
        "labels": meta.labels or {},
        "annotations": meta.annotations or {},
        "kind": obj.kind,
    }


def _workload_spec(obj) -> Dict:
    spec = obj.spec
    template = spec.template
    pod_spec = template.spec

    containers = []
    for c in pod_spec.containers:
        containers.append(
            {
                "name": c.name,
                "image": c.image,
                "resources": {
                    "limits": (c.resources.limits or {}),
                    "requests": (c.resources.requests or {}),
                }
                if c.resources
                else {"limits": {}, "requests": {}},
                "securityContext": c.security_context.to_dict()
                if getattr(c, "security_context", None)
                else {},
                "livenessProbe": c.liveness_probe.to_dict()
                if getattr(c, "liveness_probe", None)
                else None,
                "readinessProbe": c.readiness_probe.to_dict()
                if getattr(c, "readiness_probe", None)
                else None,
            }
        )

    pod_sec = pod_spec.security_context.to_dict() if pod_spec.security_context else {}

    return {
        "replicas": getattr(spec, "replicas", None),
        "containers": containers,
        "podSecurityContext": pod_sec,
        "raw": obj.to_dict(),
    }


def collect_workloads() -> List[Dict]:
    """
    Returns a list of dicts like:
      { "metadata": {...}, "spec": {...}, "kind": "Deployment" }
    """
    k8s = get_k8s_client()
    workloads: List[Dict] = []

    for dep in k8s.list_deployments().items:
        workloads.append(
            {
                "metadata": _workload_metadata(dep),
                "spec": _workload_spec(dep),
                "kind": "Deployment",
            }
        )

    for st in k8s.list_statefulsets().items:
        workloads.append(
            {
                "metadata": _workload_metadata(st),
                "spec": _workload_spec(st),
                "kind": "StatefulSet",
            }
        )

    for ds in k8s.list_daemonsets().items:
        workloads.append(
            {
                "metadata": _workload_metadata(ds),
                "spec": _workload_spec(ds),
                "kind": "DaemonSet",
            }
        )

    return workloads
