from __future__ import annotations

from typing import Dict, List

from core.checks.base import CheckContext, CheckRunResult, NullLogger
from integrations.k8s.collectors import collect_workloads


class K8sMisconfigCheck:
    """Static Kubernetes best-practice checks that tolerate missing clusters."""

    id = "k8s_misconfig"
    name = "Kubernetes misconfiguration"
    description = "Detects common Kubernetes security and reliability issues."

    async def run(self, context: CheckContext | None = None, logger: NullLogger | None = None) -> CheckRunResult:
        logger = logger or NullLogger()
        _ = context
        workloads = collect_workloads()

        issues: List[Dict] = []
        errors: List[str] = []

        for wl in workloads:
            meta: Dict = wl.get("metadata", {})
            spec: Dict = wl.get("spec", {})
            pod_sec: Dict = spec.get("podSecurityContext") or {}
            ns = meta.get("namespace", "default")
            name = meta.get("name", "unknown")
            kind = wl.get("kind", "Workload")

            resource_id = f"{kind}/{ns}/{name}"

            if not pod_sec:
                issues.append({
                    "id": f"{resource_id}:pod-security",  # lightweight placeholder
                    "resource": resource_id,
                    "message": "Pod securityContext is missing",
                })

            for container in spec.get("containers", []):
                cname = container.get("name", "container")
                cres_id = f"{resource_id}/container/{cname}"

                resources = container.get("resources") or {}
                has_limits = bool(resources.get("limits") or resources.get("requests"))
                if not has_limits:
                    issues.append({
                        "id": f"{cres_id}:resources",
                        "resource": cres_id,
                        "message": "Container resources (requests/limits) not set",
                    })

                probes = container.get("livenessProbe") or container.get("readinessProbe")
                if not probes:
                    issues.append({
                        "id": f"{cres_id}:probes",
                        "resource": cres_id,
                        "message": "Liveness/Readiness probes not configured",
                    })

                c_sec = container.get("securityContext") or {}
                if c_sec.get("privileged") is True:
                    issues.append({
                        "id": f"{cres_id}:privileged",
                        "resource": cres_id,
                        "message": "Container runs privileged",
                    })

        logger.info("k8s_misconfig scanned %s workloads", len(workloads))
        metrics = {"workloads_scanned": len(workloads), "issues_found": len(issues)}

        return CheckRunResult(issues=issues, metrics=metrics, errors=errors)


async def run_k8s_checks(context: CheckContext | None = None) -> CheckRunResult:
    return await K8sMisconfigCheck().run(context)
