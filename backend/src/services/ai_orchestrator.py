from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.checks.base import CheckContext, CheckRunResult
from core.checks.ci_hardening import CIHardeningCheck
from core.checks.k8s_misconfig import K8sMisconfigCheck
from integrations.security.scanners import ScanFinding, SecurityScannerClient
from integrations.targets.code_client import CodeIssue, CodeTargetClient
from integrations.targets.database_client import DatabaseTargetClient
from rag.SearchOrchestrator import SearchOrchestrator


@dataclass
class AnalysisRequest:
    repo_path: Optional[str] = None
    github_repo: Optional[str] = None
    database_url: Optional[str] = None
    database_schema: Optional[Dict[str, Any]] = None
    container_images: Optional[List[str]] = None
    k8s_enabled: bool = False


class AIOrchestrator:
    """High-level pipeline coordinating multi-target analysis."""

    def __init__(
        self,
        code_client: CodeTargetClient | None = None,
        db_client: DatabaseTargetClient | None = None,
        scanner: SecurityScannerClient | None = None,
        search_orchestrator: SearchOrchestrator | None = None,
    ) -> None:
        self.code_client = code_client or CodeTargetClient()
        self.db_client = db_client or DatabaseTargetClient()
        self.scanner = scanner or SecurityScannerClient()
        self.search_orchestrator = search_orchestrator or SearchOrchestrator()

    async def run(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Execute the multi-target pipeline.

        Returns a dictionary with raw results and an optional RAG summary.
        """

        if request.repo_path:
            self.code_client.repo_path = request.repo_path
        if request.database_url:
            self.db_client.connection_string = request.database_url

        ctx = CheckContext(extra={"github_repo": request.github_repo})

        code_summary = self.code_client.summarize()
        code_issues = self.code_client.collect_issues()

        db_summary = self.db_client.summarize(request.database_schema)

        check_runs: List[CheckRunResult] = []
        if request.github_repo:
            check_runs.append(await CIHardeningCheck().run(ctx))

        if request.k8s_enabled:
            check_runs.append(await K8sMisconfigCheck().run(ctx))

        scanner_findings = await self._run_scanners(request.container_images or [])

        rag_summary = await self._summarize_with_rag(
            code_summary=code_summary,
            code_issues=code_issues,
            db_summary=db_summary,
            check_runs=check_runs,
            scanner_findings=scanner_findings,
        )

        return {
            "codebase": {
                "summary": code_summary,
                "issues": [issue.__dict__ for issue in code_issues],
            },
            "database": db_summary,
            "checks": self._format_check_runs(check_runs),
            "scanners": scanner_findings,
            "rag_summary": rag_summary,
        }

    async def _run_scanners(self, images: List[str]) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        for image in images:
            image_findings = await self.scanner.scan_image(image)
            for f in image_findings:
                if isinstance(f, ScanFinding):
                    findings.append({
                        "id": f.id,
                        "title": f.title,
                        "severity": f.severity,
                        "description": f.description,
                        "source": f.source,
                        "metadata": f.metadata,
                    })
        return findings

    def _format_check_runs(self, runs: List[CheckRunResult]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []

        for run in runs:
            for issue in run.issues:
                formatted.append(
                    {
                        "check_id": issue.check_name,
                        "title": issue.title,
                        "description": issue.description or issue.short_description,
                        "severity": str(issue.severity),
                        "metadata": issue.extra,
                        "detected_at": issue.first_seen_at.isoformat()
                        if isinstance(issue.first_seen_at, datetime)
                        else issue.first_seen_at,
                    }
                )

            if not run.issues and run.metrics.get("skipped"):
                formatted.append(
                    {
                        "check_id": "n/a",
                        "title": "Check skipped",
                        "description": run.metrics.get("reason", "unknown"),
                        "severity": "info",
                        "metadata": run.metrics,
                        "detected_at": None,
                    }
                )

        return formatted

    async def _summarize_with_rag(
        self,
        *,
        code_summary: Dict[str, Any],
        code_issues: List[CodeIssue],
        db_summary: Dict[str, Any],
        check_runs: List[CheckRunResult],
        scanner_findings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Push the structured results through a lightweight RAG layer to produce
        a human-friendly summary. Any errors are swallowed so that the base
        analysis remains available even without LLM configuration.
        """

        doc_lines: List[str] = ["Codebase summary:", str(code_summary)]
        if code_issues:
            doc_lines.append("Notable inline issues:")
            for issue in code_issues[:10]:
                doc_lines.append(f"- {issue.path}:{issue.line} -> {issue.message}")

        doc_lines.append("Database summary:")
        doc_lines.append(str(db_summary))

        if check_runs:
            doc_lines.append("Check results:")
            for run in check_runs:
                if run.issues:
                    for issue in run.issues:
                        doc_lines.append(
                            f"- [{issue.severity}] {issue.title} :: {issue.description or issue.short_description}"
                        )
                elif run.metrics.get("skipped"):
                    doc_lines.append(f"- [info] Check skipped :: {run.metrics.get('reason', 'unknown')}")

        if scanner_findings:
            doc_lines.append("Scanner findings:")
            for finding in scanner_findings:
                doc_lines.append(
                    f"- ({finding.get('severity')}) {finding.get('title')} from {finding.get('source')}"
                )

        try:
            combined_text = "\n".join(doc_lines)
            self.search_orchestrator.upsert_doc(
                id="analysis-snapshot",
                text=combined_text,
                metadata={"source": "analysis"},
            )
            answer = await self.search_orchestrator.query(
                "Summarize the security posture across code, CI, Kubernetes, and dependencies.",
                intent="analysis_summary",
            )
            return {
                "answer": answer.answer,
                "mode": answer.mode,
                "intent": answer.intent,
                "citations": answer.citations,
                "latency_ms": answer.latency_ms,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "answer": "RAG summary unavailable.",
                "error": str(exc),
            }
