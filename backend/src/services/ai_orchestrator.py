import os
from integrations.targets.code_client import CodeClient
from rag.llm_client import llm_client


class AIOrchestrator:
    async def suggest_fixes(self, repo_path: str):
        code = CodeClient(repo_path)
        issues = code.run_static_analysis()

        prompt = f"""
        Analyze these code issues and generate:
        - exact fixes for each issue
        - new code blocks
        - explanation of changes

        ISSUES:
        {issues}
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

        return await llm_client.ask(prompt)

    async def auto_repair(self, repo_path: str, confirm_callback):
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
        Only applies a fix if the user confirms.
        """

        suggestions = await self.suggest_fixes(repo_path)

        apply = await confirm_callback(suggestions)
        if not apply:
            return {"status": "aborted", "message": "User rejected auto-fix."}

        return {"status": "success", "applied_changes": suggestions}
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


aio_orchestrator = AIOrchestrator()
