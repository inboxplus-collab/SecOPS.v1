import json
import os
import platform
import subprocess
from typing import Any, Dict, List

from integrations.targets.code_client import CodeClient
from integrations.targets.database_client import DatabaseClient
from integrations.security.scanners import SecurityScanner
from rag.llm_client import llm_client
from utils.errors import AnalyzerError


class AnalysisService:
    """End-to-end analyzer covering code, dependencies, and databases."""

    async def analyze_system(self, target_path: str) -> Dict[str, Any]:
        if not os.path.exists(target_path):
            raise AnalyzerError(f"Invalid target path: {target_path}")

        report: Dict[str, Any] = {
            "summary": self._system_summary(),
            "issues": [],
            "dependencies": [],
            "security_alerts": [],
            "recommendations": "",
        }

        code_client = CodeClient(target_path)
        db_client = DatabaseClient()
        scanner = SecurityScanner()

        # 1) Static analysis
        report["issues"] += code_client.run_static_analysis()

        # 2) Dependency extraction and scanning
        dependencies = code_client.extract_dependencies()
        report["dependencies"] = dependencies
        report["security_alerts"] += await scanner.check_dependencies_osv(dependencies)
        report["security_alerts"] += scanner.run_pip_audit()

        # npm audit only if package.json exists
        pkg_json = os.path.join(target_path, "package.json")
        if os.path.exists(pkg_json):
            report["security_alerts"] += scanner.run_npm_audit(target_path)

        # 3) Database inspection
        report["issues"] += db_client.scan_databases()

        # 4) LLM reasoning layer
        prompt = self._build_prompt(report)
        try:
            report["recommendations"] = await llm_client.ask(prompt)
        except Exception:
            report["recommendations"] = "LLM engine unavailable."

        return report

    # ------------------------------------------------------------------
    def _system_summary(self) -> Dict[str, Any]:
        """Collect lightweight host metadata for context."""

        summary = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cwd": os.getcwd(),
        }

        try:
            summary["git_root"] = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], text=True
            ).strip()
        except Exception:
            summary["git_root"] = None

        return summary

    def _build_prompt(self, report: Dict[str, Any]) -> str:
        return f"""
You are SecOpsAI. Analyze the following collected scan results and produce:
- Risk evaluation
- Specific fixes
- Maintenance suggestions
- DevSecOps best practices
- Step-by-step mitigation plans

SCAN DATA:
{json.dumps(report, indent=2)}
"""


analysis_service = AnalysisService()
