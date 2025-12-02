from backend.src.extensions.auto_updater.repo_client import repo_client
from backend.src.extensions.auto_updater.diff_engine import diff_engine
from backend.src.extensions.auto_updater.analyzer_client import analyzer_client
from backend.src.extensions.auto_updater.patch_engine import patch_engine
from backend.src.extensions.auto_updater.validator import validator


class AutoUpdaterAgent:

    async def update_file(self, path: str) -> dict:
        """
        Full pipeline:
        1. Pull file
        2. Analyze
        3. Generate patch
        4. Validate patch
        5. Commit to repo
        """

        # 1. Fetch
        old_content, sha = await repo_client.get_file(path)

        # 2. Analyze
        analysis = await analyzer_client.analyze_file(old_content, path)

        issues = analysis["static_issues"]
        issues.append(analysis["ai_report"])

        if not issues:
            return {"updated": False, "reason": "No issues found"}

        # 3. Patch
        patched = await patch_engine.generate_patch(path, old_content, issues)

        # 4. Validate
        if not validator.validate_python(patched):
            return {"updated": False, "reason": "AI patch invalid"}

        # 5. Commit patch
        diff = diff_engine.generate_diff(old_content, patched, path)

        await repo_client.push_file(
            path,
            patched,
            sha,
            message=f"[SecOpsAI] Auto-fix: {path}"
        )

        return {
            "updated": True,
            "diff": diff,
            "issues_found": issues
        }


auto_updater = AutoUpdaterAgent()
