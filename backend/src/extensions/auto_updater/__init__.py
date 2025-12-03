"""Auto-updater package with AI-assisted patching utilities."""

from .analyzer_client import AnalyzerClient, analyzer_client
from .diff_engine import DiffEngine, diff_engine
from .patch_engine import PatchEngine, patch_engine
from .repo_client import RepoClient, repo_client
from .validator import PatchValidator, validator

__all__ = [
    "AnalyzerClient",
    "DiffEngine",
    "PatchEngine",
    "RepoClient",
    "PatchValidator",
    "analyzer_client",
    "diff_engine",
    "patch_engine",
    "repo_client",
    "validator",
]
