"""
Enterprise-grade vulnerability scanner integration.

Features:
- OSV.dev dependency vulnerability lookup
- NVD CVE search (keyword-based, optional API key)
- Semgrep security rules execution on a codebase
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Dict, List, Optional

import httpx


OSV_URL = "https://api.osv.dev/v1/query"
NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"


class SecurityScanner:
    """
    Unified interface to all security scans.
    """

    # ========= DEPENDENCY VULNS (OSV) =========

    async def check_dependencies_osv(self, deps: List[Dict]) -> List[Dict]:
        """
        Query OSV.dev for vulnerabilities in dependencies.

        deps: list of dicts like:
          { "name": "requests", "version": "2.31.0", "ecosystem": "PyPI" }
        """
        alerts: List[Dict] = []

        async with httpx.AsyncClient(timeout=30) as client:
            for dep in deps:
                payload = {
                    "package": {
                        "name": dep["name"],
                        "ecosystem": dep["ecosystem"],
                    },
                    "version": dep.get("version"),
                }
                resp = await client.post(OSV_URL, json=payload)
                resp.raise_for_status()
                data = resp.json()

                for v in data.get("vulns", []):
                    alerts.append(
                        {
                            "source": "osv",
                            "dependency": dep["name"],
                            "ecosystem": dep["ecosystem"],
                            "id": v.get("id"),
                            "summary": v.get("summary"),
                            "aliases": v.get("aliases", []),
                            "severity": v.get("severity", []),
                            "details": v.get("details", "")[:500],
                        }
                    )
        return alerts

    # ========= DEPENDENCY VULNS (NVD) =========

    async def check_dependencies_nvd(
        self, deps: List[Dict], max_cves_per_dep: int = 5
    ) -> List[Dict]:
        """
        Very simple NVD lookup using keyword search on the package name.

        NOTE: This is heuristic – real CPE mapping is complex.
        If NVD_API_KEY is set, it will be used, otherwise unauthenticated (rate-limited).
        """
        api_key = os.getenv("NVD_API_KEY")
        headers = {}
        if api_key:
            headers["apiKey"] = api_key

        vulns: List[Dict] = []
        async with httpx.AsyncClient(timeout=30) as client:
            for dep in deps:
                params = {
                    "keywordSearch": dep["name"],
                    "resultsPerPage": max_cves_per_dep,
                }
                resp = await client.get(NVD_URL, params=params, headers=headers)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                for item in data.get("vulnerabilities", []):
                    cve = item.get("cve", {})
                    vulns.append(
                        {
                            "source": "nvd",
                            "dependency": dep["name"],
                            "id": cve.get("id"),
                            "summary": cve.get("descriptions", [{}])[0].get("value", ""),
                            "metrics": cve.get("metrics", {}),
                            "published": cve.get("published"),
                            "lastModified": cve.get("lastModified"),
                        }
                    )
        return vulns

    # ========= CODE VULNS (SEMGREP) =========

    def run_semgrep(
        self,
        target_path: str,
        ruleset: str = "p/ci",
    ) -> List[Dict]:
        """
        Run Semgrep with a security-focused ruleset.

        - requires `semgrep` to be installed in the environment.
        - `ruleset` can be:
            - "p/ci" (fast CI rules)
            - "p/security-audit"
            - a local rules folder

        Returns a list of vulnerability findings.
        """
        findings: List[Dict] = []

        try:
            cmd = [
                "semgrep",
                "--config",
                ruleset,
                target_path,
                "--json",
            ]
            out = subprocess.check_output(cmd, text=True)
            data = json.loads(out)

            for result in data.get("results", []):
                extra = result.get("extra", {})
                findings.append(
                    {
                        "source": "semgrep",
                        "rule_id": extra.get("rule_id"),
                        "severity": extra.get("severity"),
                        "message": extra.get("message"),
                        "path": result.get("path"),
                        "start": result.get("start", {}),
                        "end": result.get("end", {}),
                        "metavars": extra.get("metavars", {}),
                    }
                )

        except FileNotFoundError:
            # semgrep not installed; we fail gracefully
            findings.append(
                {
                    "source": "semgrep",
                    "error": "semgrep_not_installed",
                    "message": "semgrep is not installed in the runtime environment.",
                }
            )
        except subprocess.CalledProcessError as e:
            findings.append(
                {
                    "source": "semgrep",
                    "error": "semgrep_failed",
                    "message": str(e),
                }
            )

        return findings

    # ========= HIGH-LEVEL AGGREGATOR =========

    async def full_vulnerability_report(
        self,
        deps: List[Dict],
        code_path: Optional[str] = None,
        semgrep_ruleset: str = "p/ci",
    ) -> Dict:
        """
        High-level API combining:
        - OSV.dev dependency scan
        - NVD keyword-based CVE lookup
        - Semgrep static analysis on codebase

        Returns a dict with 'dependency_vulns', 'code_vulns', 'stats'
        """
        osv_vulns = await self.check_dependencies_osv(deps)
        nvd_vulns = await self.check_dependencies_nvd(deps)

        semgrep_vulns: List[Dict] = []
        if code_path:
            semgrep_vulns = self.run_semgrep(code_path, ruleset=semgrep_ruleset)

        return {
            "dependency_vulns": osv_vulns + nvd_vulns,
            "code_vulns": semgrep_vulns,
            "stats": {
                "osv_count": len(osv_vulns),
                "nvd_count": len(nvd_vulns),
                "semgrep_count": len(semgrep_vulns),
            },
        }


security_scanner = SecurityScanner()
