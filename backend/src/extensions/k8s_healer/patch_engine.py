"""
Translates k8s misconfiguration CheckResult into JSON patch bodies.
We keep patches small and additive so they are low-risk.
"""

from __future__ import annotations

from typing import Dict
from backend.src.core.checks.base import CheckResult


class K8sPatchEngine:

    def build_patch_for_issue(self, issue: CheckResult) -> Dict:
        """
        Returns a strategic merge patch dict to be applied
        to a Deployment/StatefulSet/DaemonSet.
        """

        # resource looks like: kind/namespace/name[/container/foo]
        parts = issue.resource.split("/")
        kind = parts[0]          # Deployment, StatefulSet, DaemonSet
        namespace = parts[1]
        name = parts[2]

        container_name = parts[4] if len(parts) > 4 else None

        base = {
            "kind": kind,
            "metadata": {"name": name, "namespace": namespace},
        }

        if issue.check_id == "k8s_pod_security_context_missing":
            return {
                **base,
                "spec": {
                    "template": {
                        "spec": {
                            "securityContext": {
                                "runAsNonRoot": True,
                                "seccompProfile": {"type": "RuntimeDefault"},
                            }
                        }
                    }
                },
            }

        if issue.check_id == "k8s_container_resources_missing" and container_name:
            return {
                **base,
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": container_name,
                                    "resources": {
                                        "requests": {"cpu": "100m", "memory": "128Mi"},
                                        "limits": {"cpu": "500m", "memory": "512Mi"},
                                    },
                                }
                            ]
                        }
                    }
                },
            }

        if issue.check_id == "k8s_container_probes_missing" and container_name:
            # Generic HTTP probe on /healthz
            return {
                **base,
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": container_name,
                                    "livenessProbe": {
                                        "httpGet": {"path": "/healthz", "port": 80},
                                        "initialDelaySeconds": 30,
                                        "timeoutSeconds": 5,
                                    },
                                    "readinessProbe": {
                                        "httpGet": {"path": "/healthz", "port": 80},
                                        "initialDelaySeconds": 5,
                                        "timeoutSeconds": 3,
                                    },
                                }
                            ]
                        }
                    }
                },
            }

        if issue.check_id == "k8s_container_privileged" and container_name:
            return {
                **base,
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": container_name,
                                    "securityContext": {
                                        "privileged": False,
                                        "runAsNonRoot": True,
                                        "allowPrivilegeEscalation": False,
                                    },
                                }
                            ]
                        }
                    }
                },
            }

        # Fallback: no patch
        return base


patch_engine = K8sPatchEngine()
