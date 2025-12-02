from __future__ import annotations

import json
import subprocess
from typing import Dict, List

import httpx


class SecurityScanner:
    """Multi-source dependency and container scanner.

    Combines OSV lookups with local package audits (pip-audit, npm audit) to
    provide actionable vulnerability signals. Each tool is optional; failures
    degrade gracefully without interrupting the calling workflow.
    """

    async def check_dependencies_osv(self, deps: List[Dict]) -> List[Dict]:
        alerts: List[Dict] = []
        url = "https://api.osv.dev/v1/query"

        async with httpx.AsyncClient(timeout=30) as client:
            for dep in deps:
                payload = {
                    "package": {"name": dep.get("name"), "ecosystem": dep.get("ecosystem")},
                    "version": dep.get("version"),
                }
                try:
                    resp = await client.post(url, json=payload)
                    data = resp.json()
                except Exception:
                    continue

                for vuln in data.get("vulns", []) or []:
                    alerts.append(
                        {
                            "dependency": dep.get("name"),
                            "id": vuln.get("id"),
                            "summary": vuln.get("summary", ""),
                            "severity": vuln.get("severity", "unknown"),
                            "source": "osv",
                        }
                    )
        return alerts

    def run_pip_audit(self) -> List[Dict]:
        """Run pip-audit against the current environment/requirements."""

        findings: List[Dict] = []
        try:
            raw = subprocess.check_output(["pip-audit", "-f", "json"], text=True)
            data = json.loads(raw)
            for item in data:
                vuln = item.get("vulns", [{}])[0] if item.get("vulns") else {}
                findings.append(
                    {
                        "dependency": item.get("name"),
                        "version": item.get("version"),
                        "id": vuln.get("id"),
                        "fix_versions": vuln.get("fix_versions", []),
                        "severity": (vuln.get("severity") or "unknown").lower(),
                        "summary": vuln.get("description", ""),
                        "source": "pip-audit",
                    }
                )
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            return findings
        return findings

    def run_npm_audit(self, manifest_dir: str) -> List[Dict]:
        """Execute `npm audit --json` within the provided manifest directory."""

        findings: List[Dict] = []
        try:
            raw = subprocess.check_output(["npm", "audit", "--json"], cwd=manifest_dir, text=True)
            data = json.loads(raw)
            for advisory in (data.get("vulnerabilities") or {}).values():
                findings.append(
                    {
                        "dependency": advisory.get("name"),
                        "severity": advisory.get("severity"),
                        "summary": advisory.get("via"),
                        "source": "npm-audit",
                    }
                )
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            return findings
        return findings
