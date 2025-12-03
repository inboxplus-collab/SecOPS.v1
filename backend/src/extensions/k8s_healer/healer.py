"""
Kubernetes auto-healer engine.

Takes misconfiguration check results and:
- Builds strategic merge patches
- Applies them via Kubernetes API (if apply=True)
"""

from __future__ import annotations

from typing import Dict, List

from backend.src.core.checks.k8s_misconfig import run_k8s_checks
from backend.src.integrations.k8s.client import get_k8s_client
from backend.src.extensions.k8s_healer.patch_engine import patch_engine
from backend.src.core.checks.base import CheckResult


class K8sHealer:

    async def analyze(self) -> Dict:
        """
        Run k8s misconfig checks and return issues + suggested patches (no apply).
        """
        issues: List[CheckResult] = run_k8s_checks()
        patches: List[Dict] = [patch_engine.build_patch_for_issue(i) for i in issues]

        return {
            "issues": [i.dict() for i in issues],
            "patches": patches,
            "summary": {
                "issue_count": len(issues),
                "patch_count": len(patches),
            },
        }

    async def heal(self, apply: bool = False) -> Dict:
        """
        If apply=False → only simulation (dry run).
        If apply=True  → actually patch cluster workloads.
        """
        analysis = await self.analyze()

        if not apply:
            analysis["applied"] = False
            return analysis

        k8s = get_k8s_client()
        applied = []

        for patch in analysis["patches"]:
            kind = patch["kind"]
            name = patch["metadata"]["name"]
            ns = patch["metadata"]["namespace"]

            if kind == "Deployment":
                k8s.patch_deployment(name, ns, patch)
            elif kind == "StatefulSet":
                k8s.patch_statefulset(name, ns, patch)
            elif kind == "DaemonSet":
                k8s.patch_daemonset(name, ns, patch)
            else:
                # unknown kind; skip
                continue

            applied.append(f"{kind}/{ns}/{name}")

        analysis["applied"] = True
        analysis["applied_resources"] = applied
        return analysis


k8s_healer = K8sHealer()
