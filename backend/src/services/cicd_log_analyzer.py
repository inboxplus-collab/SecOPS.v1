from __future__ import annotations

from typing import Any, Dict, Optional

from integrations.ci.github_actions import GitHubActionsClient
from integrations.ci.jenkins import jenkins_client
from rag.llm_client import llm_client
from extensions.auto_updater.repo_client import repo_client


class CICDLogAnalyzerService:
    """High-level CI/CD log analyzer with GitHub Actions and Jenkins helpers."""

    def __init__(self) -> None:
        self._github_client: GitHubActionsClient | None = None

    def _get_github_client(self) -> GitHubActionsClient:
        if self._github_client is None:
            self._github_client = GitHubActionsClient.from_env()
        return self._github_client

    async def analyze_github_run(
        self,
        run_id: Optional[int] = None,
        branch: Optional[str] = None,
    ) -> Dict[str, Any]:
        client = self._get_github_client()

        run = None
        if run_id is not None:
            runs = await client.list_workflow_runs(branch=branch, per_page=20)
            for candidate in runs:
                if candidate.get("id") == run_id:
                    run = candidate
                    break
        else:
            run = await client.get_latest_failed_run(branch=branch)

        if not run:
            return {"found": False, "reason": "No failed workflow run found."}

        logs_summary = await client.summarize_run_logs(run["id"])

        prompt = f"""
You are an expert CI/CD engineer.

A GitHub Actions workflow has failed.

WORKFLOW RUN METADATA:
{run}

JOB & STEP LOG SUMMARY:
{logs_summary}

TASK:
1. Explain in plain language WHY the pipeline failed.
2. Identify the most likely root cause (config vs code vs environment).
3. Suggest a concrete fix.
4. If it's a workflow (YAML) issue, propose changes clearly.
5. If it's a code issue, describe what needs to be changed and where.
Return your answer in a structured, human-readable format.
"""

        explanation = await llm_client.ask(prompt)

        return {
            "found": True,
            "run": run,
            "logs_summary": logs_summary,
            "analysis": explanation,
        }

    async def auto_fix_github_workflow(
        self,
        workflow_path: str,
        run_id: Optional[int] = None,
        branch: Optional[str] = None,
    ) -> Dict[str, Any]:
        base_analysis = await self.analyze_github_run(run_id=run_id, branch=branch)
        if not base_analysis.get("found"):
            return {**base_analysis, "patched": False}

        try:
            current_yaml, sha = await repo_client.get_file(workflow_path)
        except Exception as exc:  # noqa: BLE001
            return {
                "found": True,
                "patched": False,
                "error": f"Failed to fetch workflow file: {exc}",
                "analysis": base_analysis.get("analysis"),
            }

        prompt = f"""
You are an expert in GitHub Actions YAML.

The following workflow file is associated with a failing run.

CURRENT WORKFLOW YAML ({workflow_path}):
---
{current_yaml}
---

RUN FAILURE ANALYSIS:
{base_analysis.get("analysis")}

JOB & STEP LOG SUMMARY:
{base_analysis.get("logs_summary")}

TASK:
Rewrite the FULL workflow YAML so that:
- It fixes the failure cause if it is config-related.
- It remains logically equivalent where things already work.
- It is valid GitHub Actions YAML.
Output ONLY the updated YAML, no explanation, no markdown, no backticks.
"""

        fixed_yaml = await llm_client.ask(prompt)

        await repo_client.push_file(
            workflow_path,
            fixed_yaml,
            sha,
            message=f"[T79AI] Auto-fix GitHub Actions workflow: {workflow_path}",
        )

        return {
            "found": True,
            "patched": True,
            "workflow_path": workflow_path,
            "analysis": base_analysis.get("analysis"),
        }

    async def analyze_jenkins_job(self, job_name: str) -> Dict[str, Any]:
        try:
            build = await jenkins_client.get_latest_failed_build(job_name)
        except RuntimeError as exc:
            return {"found": False, "reason": str(exc)}

        if not build:
            return {"found": False, "reason": "No failed builds found for this job."}

        build_url = build.get("url")
        console = await jenkins_client.get_console_log(build_url)

        prompt = f"""
You are an expert Jenkins pipeline engineer.

A Jenkins job has a failing build.

JOB NAME: {job_name}
BUILD URL: {build_url}

CONSOLE LOG:
{console[:15000]}

TASK:
1. Summarize why the build failed.
2. Identify the most likely root cause (pipeline config vs code vs environment).
3. Suggest exact changes to Jenkinsfile or project config if relevant.
4. Suggest how to re-run and verify the fix.

Answer clearly in structured bullet points.
"""

        explanation = await llm_client.ask(prompt)

        return {
            "found": True,
            "job": job_name,
            "build": {"url": build_url, "number": build.get("number")},
            "analysis": explanation,
        }


cicd_log_analyzer = CICDLogAnalyzerService()
