from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from typing import Dict, List


class CodeClient:
    """Realistic code-scanning helper for repositories.

    The client wraps a handful of battle-tested linters and scanners while
    defending against missing tools or noisy failures. All operations are
    scoped to ``repo_path`` to avoid accidental modification outside the
    workspace.
    """

    def __init__(self, repo_path: str) -> None:
        self.repo_path = os.path.abspath(repo_path)

    # ------------------------------------------------------------------
    # Static analysis
    # ------------------------------------------------------------------
    def _run_command(self, args: List[str]) -> str | None:
        try:
            return subprocess.check_output(args, text=True, cwd=self.repo_path)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def run_static_analysis(self) -> List[Dict]:
        """
        Run Bandit, Semgrep, and pylint, merging their results into a unified
        issue list. Each tool is optional—if it's not available locally the
        failure is swallowed so the pipeline can continue.
        """

        issues: List[Dict] = []

        # Bandit
        bandit_out = self._run_command(["bandit", "-r", self.repo_path, "-f", "json"])
        if bandit_out:
            try:
                data = json.loads(bandit_out)
                for item in data.get("results", []):
                    issues.append(
                        {
                            "tool": "bandit",
                            "severity": item.get("issue_severity", "UNKNOWN"),
                            "issue": item.get("issue_text", ""),
                            "file": item.get("filename"),
                            "line": item.get("line_number"),
                        }
                    )
            except json.JSONDecodeError:
                pass

        # Semgrep
        semgrep_out = self._run_command(["semgrep", "--config", "auto", self.repo_path, "--json"])
        if semgrep_out:
            try:
                data = json.loads(semgrep_out)
                for result in data.get("results", []):
                    extra = result.get("extra", {})
                    issues.append(
                        {
                            "tool": "semgrep",
                            "severity": extra.get("severity", "UNKNOWN"),
                            "issue": extra.get("message", ""),
                            "file": result.get("path"),
                            "line": result.get("start", {}).get("line"),
                        }
                    )
            except json.JSONDecodeError:
                pass

        # pylint
        pylint_out = self._run_command(["pylint", self.repo_path, "-f", "json"])
        if pylint_out:
            try:
                data = json.loads(pylint_out)
                for item in data:
                    issues.append(
                        {
                            "tool": "pylint",
                            "severity": item.get("type", "info"),
                            "issue": item.get("message", ""),
                            "file": item.get("path"),
                            "line": item.get("line"),
                        }
                    )
            except json.JSONDecodeError:
                pass

        return issues

    # ------------------------------------------------------------------
    # Dependency extraction
    # ------------------------------------------------------------------
    def extract_dependencies(self) -> List[Dict]:
        """Extract runtime dependencies from common manifest files."""

        deps: List[Dict] = []

        req = os.path.join(self.repo_path, "requirements.txt")
        if os.path.exists(req):
            with open(req, encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "==" in line:
                        name, version = line.split("==", 1)
                    else:
                        name, version = line, "latest"
                    deps.append({"name": name, "version": version, "ecosystem": "PyPI"})

        pkg_json = os.path.join(self.repo_path, "package.json")
        if os.path.exists(pkg_json):
            try:
                data = json.loads(open(pkg_json, encoding="utf-8").read())
            except json.JSONDecodeError:
                data = {}
            for name, version in data.get("dependencies", {}).items():
                deps.append({"name": name, "version": version, "ecosystem": "npm"})
            for name, version in data.get("devDependencies", {}).items():
                deps.append({"name": name, "version": version, "ecosystem": "npm"})

        return deps

    # ------------------------------------------------------------------
    # Safe file operations
    # ------------------------------------------------------------------
    def rewrite_file(self, relative_path: str, new_content: str) -> str:
        """
        Safely rewrite a file under the repository, creating a backup copy.

        Returns the path to the backup for auditability.
        """

        target = os.path.abspath(os.path.join(self.repo_path, relative_path))
        if not target.startswith(self.repo_path):
            raise ValueError("Attempted to write outside of repository root")

        os.makedirs(os.path.dirname(target), exist_ok=True)
        backup = f"{target}.bak"
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
            tmp.write(new_content)
            temp_path = tmp.name

        if os.path.exists(target):
            shutil.copy2(target, backup)
        os.replace(temp_path, target)
        return backup
