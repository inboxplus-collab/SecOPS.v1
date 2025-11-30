from __future__ import annotations

from datetime import datetime
from typing import List

from .base import BaseCheck, CheckContext, CheckResult
from integrations.k8s.client import get_k8s_apps_api, get_k8s_core_api


class K8sMisconfigCheck(BaseCheck):
    id = "k8s_misconfig"
    name = "Kubernetes misconfiguration"
    description = "Checks basic K8s best practices: resource limits, public services, etc."

    async def run(self, ctx: CheckContext) -> List[CheckResult]:
        apps_api = get_k8s_apps_api()
        core_api = get_k8s_core_api()
        results: List[CheckResult] = []

        # NOTE: client is sync; we call it in thread in real world, but for simplicity we treat as sync here.
        deployments = apps_api.list_deployment_for_all_namespaces().items
        for dep in deployments:
            ns = dep.metadata.namespace
            name = dep.metadata.name
            for c in dep.spec.template.spec.containers:
                if not (c.resources and c.resources.limits):
                    results.append(
                        CheckResult(
                            check_id=self.id,
                            title=f"K8s deployment without resource limits: {ns}/{name}",
                            description="Container has no resource limits set. This can lead to noisy neighbors and instability.",
                            severity="medium",
                            metadata={
                                "namespace": ns,
                                "deployment": name,
                                "container": c.name,
                            },
                            detected_at=datetime.utcnow(),
                        )
                    )

        # Public LoadBalancer services
        services = core_api.list_service_for_all_namespaces().items
        for svc in services:
            if svc.spec.type == "LoadBalancer" and not svc.spec.load_balancer_source_ranges:
                results.append(
                    CheckResult(
                        check_id=self.id,
                        title=f"Public LoadBalancer service: {svc.metadata.namespace}/{svc.metadata.name}",
                        description="Service of type LoadBalancer without source ranges can be exposed to the whole internet.",
                        severity="high",
                        metadata={
                            "namespace": svc.metadata.namespace,
                            "service": svc.metadata.name,
                        },
                        detected_at=datetime.utcnow(),
                    )
                )

        return results
