from __future__ import annotations

from datetime import datetime
from typing import Any, List

from api.schemas.issues import IssueSeverity, IssueStatus, IssueLocation
from .base import BaseCheck, CheckContext, CheckRunResult, LoggerLike, NullLogger
from integrations.k8s.client import get_k8s_apps_api, get_k8s_core_api


class K8sMisconfigCheck(BaseCheck):
    id = "k8s_misconfig"
    name = "Kubernetes misconfiguration"
    description = "Checks basic K8s best practices: resource limits, public services, etc."

    def __init__(self) -> None:
        super().__init__(default_severity=IssueSeverity.medium)

    async def run(
        self, context: CheckContext, logger: LoggerLike | None = None
    ) -> CheckRunResult:
        logger = logger or NullLogger()
        issues: List[Any] = []
        errors: List[str] = []

        try:
            apps_api = get_k8s_apps_api()
            core_api = get_k8s_core_api()
        except Exception as exc:  # noqa: BLE001
            logger.warning("K8sMisconfigCheck skipped: %s", exc)
            return CheckRunResult(
                issues=[],
                metrics={"skipped": True, "reason": "k8s_client_error"},
                errors=[str(exc)],
            )

        try:
            deployments = apps_api.list_deployment_for_all_namespaces().items
        except Exception as exc:  # noqa: BLE001
            logger.warning("K8sMisconfigCheck could not list deployments: %s", exc)
            deployments = []
            errors.append(str(exc))

        for dep in deployments:
            ns = dep.metadata.namespace
            name = dep.metadata.name
            for c in dep.spec.template.spec.containers:
                if not (c.resources and c.resources.limits):
                    location = IssueLocation(
                        kind="k8s",
                        namespace=ns,
                        resource_kind="Deployment",
                        resource_name=name,
                        metadata={"container": c.name},
                    )
                    issues.append(
                        self.build_issue(
                            id=f"{context.org_id}:{ns}:{name}:{c.name}:limits",
                            org_id=context.org_id or "unknown-org",
                            title=f"K8s deployment without resource limits: {ns}/{name}",
                            severity=IssueSeverity.medium,
                            status=IssueStatus.open,
                            service=name,
                            category="k8s",
                            tags=["k8s", "resources", "hygiene"],
                            location=location,
                            source="k8s",
                            short_description="Container has no resource limits set.",
                            description="Container has no resource limits set. This can lead to noisy neighbors and instability.",
                            root_cause="Resource limits were omitted from the deployment spec.",
                            impact="Pods may consume excessive resources and disrupt cluster stability.",
                            proposed_fix="Define CPU/memory requests and limits for each container.",
                            precautions="Validate limits in staging before rollout.",
                            references=[],
                            extra={},
                        )
                    )

        try:
            services = core_api.list_service_for_all_namespaces().items
        except Exception as exc:  # noqa: BLE001
            logger.warning("K8sMisconfigCheck could not list services: %s", exc)
            services = []
            errors.append(str(exc))

        for svc in services:
            if svc.spec.type == "LoadBalancer" and not svc.spec.load_balancer_source_ranges:
                location = IssueLocation(
                    kind="k8s",
                    namespace=svc.metadata.namespace,
                    resource_kind="Service",
                    resource_name=svc.metadata.name,
                )
                issues.append(
                    self.build_issue(
                        id=f"{context.org_id}:{svc.metadata.namespace}:{svc.metadata.name}:public",
                        org_id=context.org_id or "unknown-org",
                        title=f"Public LoadBalancer service: {svc.metadata.namespace}/{svc.metadata.name}",
                        severity=IssueSeverity.high,
                        status=IssueStatus.open,
                        service=svc.metadata.name,
                        category="k8s",
                        tags=["k8s", "network", "exposure"],
                        location=location,
                        source="k8s",
                        short_description="Service is exposed via public LoadBalancer.",
                        description="Service of type LoadBalancer without source ranges can be exposed to the whole internet.",
                        root_cause="Service type set to LoadBalancer without source ranges.",
                        impact="Workloads may be reachable from untrusted networks.",
                        proposed_fix="Restrict source ranges or use ClusterIP/Ingress based exposure.",
                        precautions="Ensure ingress paths remain functional after changes.",
                        references=[],
                        extra={},
                    )
                )

        metrics = {
            "deployments_scanned": len(deployments) if 'deployments' in locals() else 0,
            "services_scanned": len(services) if 'services' in locals() else 0,
            "issues_found": len(issues),
        }

        return CheckRunResult(issues=issues, metrics=metrics, errors=errors)
