from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class _EmptyList:
    items: List = None

    def __post_init__(self) -> None:
        if self.items is None:
            self.items = []


class K8sClient:
    """Minimal stub client used for local testing."""

    def list_deployments(self, namespace: Optional[str] = None) -> _EmptyList:
        return _EmptyList()

    def list_statefulsets(self, namespace: Optional[str] = None) -> _EmptyList:
        return _EmptyList()

    def list_daemonsets(self, namespace: Optional[str] = None) -> _EmptyList:
        return _EmptyList()


def get_k8s_client() -> K8sClient:
    return K8sClient()
