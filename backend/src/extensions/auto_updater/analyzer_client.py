import ast
from backend.src.rag.llm_client import llm_client


class AnalyzerClient:

    async def analyze_file(self, code: str, filepath: str) -> dict:
        """
        Uses static AST analysis + AI semantic reasoning.
        Returns:
            {
                "issues": [...],
                "severity": "low/medium/high",
                "fix_required": bool
            }
        """

        issues = []

        # === STATIC ANALYSIS: Python AST ===
        try:
            ast.parse(code)
        except Exception as e:
            issues.append(f"SyntaxError in {filepath}: {e}")

        # AI reasoning
        prompt = f"""
You are an expert senior engineer.
Analyze the following file and find:
- bugs
- security issues
- anti-patterns
- inefficiencies

Then output JSON with:
issues: []
severity: "low/medium/high"
fix_required: true/false

CODE:
{code}
"""
        ai_result = await llm_client.ask(prompt)

        return {
            "static_issues": issues,
            "ai_report": ai_result,
        }


analyzer_client = AnalyzerClient()
