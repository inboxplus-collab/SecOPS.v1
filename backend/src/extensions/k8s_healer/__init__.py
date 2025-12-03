"""Kubernetes auto-healer extension."""

from backend.src.extensions.k8s_healer.healer import K8sHealer, k8s_healer

__all__ = ["K8sHealer", "k8s_healer"]
