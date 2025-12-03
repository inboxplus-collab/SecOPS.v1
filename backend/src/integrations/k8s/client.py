"""
Kubernetes API client wrapper.

Supports:
- In-cluster config (when running inside k8s)
- Kubeconfig file (local dev)
Falls back gracefully if kubernetes client is not installed.
"""

from __future__ import annotations

from typing import Optional

from backend.src.utils.config import settings


try:
    from kubernetes import client, config
    K8S_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    client = None
    config = None
    K8S_AVAILABLE = False


class K8sClient:
    def __init__(self):
        if not K8S_AVAILABLE:
            raise RuntimeError(
                "kubernetes python client is not installed. "
                "Install with: pip install kubernetes"
            )

        if settings.K8S_IN_CLUSTER:
            config.load_incluster_config()
        else:
            if settings.K8S_KUBECONFIG:
                config.load_kube_config(config_file=settings.K8S_KUBECONFIG)
            else:
                config.load_kube_config()

        self.apps = client.AppsV1Api()
        self.core = client.CoreV1Api()
        self.networking = client.NetworkingV1Api()

    # ========= LISTERS =========

    def list_deployments(self, namespace: Optional[str] = None):
        if namespace:
            return self.apps.list_namespaced_deployment(namespace)
        return self.apps.list_deployment_for_all_namespaces()

    def list_statefulsets(self, namespace: Optional[str] = None):
        if namespace:
            return self.apps.list_namespaced_stateful_set(namespace)
        return self.apps.list_stateful_set_for_all_namespaces()

    def list_daemonsets(self, namespace: Optional[str] = None):
        if namespace:
            return self.apps.list_namespaced_daemon_set(namespace)
        return self.apps.list_daemon_set_for_all_namespaces()

    def list_pods(self, namespace: Optional[str] = None):
        if namespace:
            return self.core.list_namespaced_pod(namespace)
        return self.core.list_pod_for_all_namespaces()

    def list_ingresses(self, namespace: Optional[str] = None):
        if namespace:
            return self.networking.list_namespaced_ingress(namespace)
        return self.networking.list_ingress_for_all_namespaces()

    # ========= PATCHERS =========

    def patch_deployment(self, name: str, namespace: str, patch_body: dict):
        return self.apps.patch_namespaced_deployment(
            name=name, namespace=namespace, body=patch_body
        )

    def patch_statefulset(self, name: str, namespace: str, patch_body: dict):
        return self.apps.patch_namespaced_stateful_set(
            name=name, namespace=namespace, body=patch_body
        )

    def patch_daemonset(self, name: str, namespace: str, patch_body: dict):
        return self.apps.patch_namespaced_daemon_set(
            name=name, namespace=namespace, body=patch_body
        )


def get_k8s_client() -> K8sClient:
    return K8sClient()
