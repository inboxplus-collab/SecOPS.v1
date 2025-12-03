"""
Kubernetes misconfiguration checks.

Catches:
- Missing CPU/memory requests/limits
- No liveness/readiness probes
- Privileged containers / runAsRoot
- No pod securityContext
"""

from __future__ import annotations

from typing import Dict, List

from backend.src.core.checks.base import CheckResult, Severity
from backend.src.integrations.k8s.collectors import collect_workloads


def _has_resources(container: Dict) -> bool:
    res = container.get("resources") or {}
    return bool(res.get("limits") or res.get("requests"))


def _has_probes(container: Dict) -> bool:
    return bool(container.get("livenessProbe") or container.get("readinessProbe"))


def _is_privileged(container: Dict, pod_sec: Dict) -> bool:
    c_sec = container.get("securityContext") or {}
    if c_sec.get("privileged") is True:
        return True

    # runAsUser 0 is effectively root
    run_as_user = c_sec.get("runAsUser") or pod_sec.get("runAsUser")
    if run_as_user == 0:
        return True

    return False


def run_k8s_checks() -> List[CheckResult]:
    workloads = collect_workloads()
    results: List[CheckResult] = []

    for wl in workloads:
        meta = wl["metadata"]
        spec = wl["spec"]
        pod_sec = spec.get("podSecurityContext") or {}
        ns = meta["namespace"]
        name = meta["name"]
        kind = wl["kind"]

        resource_id = f"{kind}/{ns}/{name}"

        # === Check 1: pod-level security context missing ===
        if not pod_sec:
            results.append(
                CheckResult(
                    check_id="k8s_pod_security_context_missing",
                    title="Pod securityContext is missing",
                    severity=Severity.medium,
                    description=(
                        f"{resource_id} has no pod-level securityContext. "
                        "This increases risk of privilege escalation and "
                        "inconsistent security posture."
                    ),
                    resource=resource_id,
                    remediation=(
                        "Define a pod-level `securityContext` with runAsNonRoot=true, "
                        "seccompProfile, and fsGroup where appropriate."
                    ),
                    metadata={"namespace": ns, "kind": kind},
                )
            )

        for c in spec.get("containers", []):
            cname = c["name"]
            cres_id = f"{resource_id}/container/{cname}"

            # === Check 2: missing requests/limits ===
            if not _has_resources(c):
                results.append(
                    CheckResult(
                        check_id="k8s_container_resources_missing",
                        title="Container resources (requests/limits) not set",
                        severity=Severity.medium,
                        description=(
                            f"{cres_id} has no CPU/memory requests or limits. "
                            "This can lead to noisy-neighbor problems or OOM kills."
                        ),
                        resource=cres_id,
                        remediation=(
                            "Set `resources.requests` and `resources.limits` for "
                            "CPU and memory to ensure fair scheduling and stability."
                        ),
                        metadata={"namespace": ns, "kind": kind},
                    )
                )

            # === Check 3: no health probes ===
            if not _has_probes(c):
                results.append(
                    CheckResult(
                        check_id="k8s_container_probes_missing",
                        title="Liveness/Readiness probes not configured",
                        severity=Severity.medium,
                        description=(
                            f"{cres_id} has no liveness or readiness probes. "
                            "Kubernetes will not detect failures or slow startups properly."
                        ),
                        resource=cres_id,
                        remediation=(
                            "Configure appropriate `livenessProbe` and `readinessProbe` "
                            "for the container, e.g. HTTP or TCP checks."
                        ),
                        metadata={"namespace": ns, "kind": kind},
                    )
                )

            # === Check 4: privileged / root ===
            if _is_privileged(c, pod_sec):
                results.append(
                    CheckResult(
                        check_id="k8s_container_privileged",
                        title="Container runs privileged or as root",
                        severity=Severity.high,
                        description=(
                            f"{cres_id} is running as privileged or as root user. "
                            "This significantly increases the blast radius if compromised."
                        ),
                        resource=cres_id,
                        remediation=(
                            "Remove `privileged: true`, set `runAsNonRoot: true`, "
                            "and use `runAsUser` with a non-zero UID."
                        ),
                        metadata={"namespace": ns, "kind": kind},
                    )
                )

    return results
