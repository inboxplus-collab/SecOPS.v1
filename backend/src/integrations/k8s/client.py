from __future__ import annotations

import os

from kubernetes import client, config


_k8s_config_loaded = False


def _ensure_config_loaded() -> None:
    global _k8s_config_loaded
    if _k8s_config_loaded:
        return

    # Prefer in-cluster config, fall back to local ~/.kube/config
    try:
        if os.getenv("KUBERNETES_SERVICE_HOST"):
            config.load_incluster_config()
        else:
            config.load_kube_config()
        _k8s_config_loaded = True
    except Exception:
        # We fail soft here; checks will simply see no data.
        _k8s_config_loaded = False


def get_k8s_apps_api() -> client.AppsV1Api:
    _ensure_config_loaded()
    return client.AppsV1Api()


def get_k8s_core_api() -> client.CoreV1Api:
    _ensure_config_loaded()
    return client.CoreV1Api()
