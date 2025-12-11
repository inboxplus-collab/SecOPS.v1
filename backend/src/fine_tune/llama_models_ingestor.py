"""Utilities to build datasets from the inboxplus-collab/llama-models repo.

Because this environment cannot always reach the public internet, the ingestor
accepts a pre-cloned repository path as an optional input. When network access
is available, it will clone the upstream repository into a temporary directory
and convert all text-like files into instruction-style prompt/completion pairs
using ``build_dataset``.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

from .dataset_builder import build_dataset

REPO_URL = "https://github.com/inboxplus-collab/llama-models.git"
DEFAULT_BRANCH = "main"
_TEXT_EXTENSIONS = (".md", ".txt", ".json", ".yaml", ".yml")


def _clone_repo(repo_url: str, branch: str | None) -> Path:
    destination = Path(tempfile.mkdtemp(prefix="llama_models_"))
    command = ["git", "clone"]
    if branch:
        command.extend(["-b", branch])
    command.extend([repo_url, str(destination)])

    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    return destination


def _iter_text_files(root: Path, extensions: Iterable[str]) -> Iterable[Path]:
    for ext in extensions:
        for path in root.rglob(f"*{ext}"):
            if any(part.startswith(".") for part in path.parts):
                continue
            if path.is_file():
                yield path


def build_llama_models_dataset(
    output_path: str,
    *,
    source_path: str | None = None,
    repo_url: str = REPO_URL,
    branch: str | None = DEFAULT_BRANCH,
    tags: Sequence[str] | None = None,
) -> Path:
    """Create a JSONL dataset from the llama-models repository contents.

    Parameters
    ----------
    output_path:
        Target file path for the JSONL dataset.
    source_path:
        Optional local checkout of the repository. If omitted, the ingestor will
        attempt to clone ``repo_url``. This is useful for offline environments
        where the repository must be provided manually.
    repo_url / branch:
        Upstream Git reference used when cloning the repository automatically.
    tags:
        Additional metadata tags to store alongside each example.
    """

    cleanup_root: Path | None = None
    repo_root = Path(source_path) if source_path else _clone_repo(repo_url, branch)
    if not source_path:
        cleanup_root = repo_root

    try:
        files = list(_iter_text_files(repo_root, _TEXT_EXTENSIONS))
        if not files:
            raise FileNotFoundError(
                f"No text-like files found under {repo_root}. Provide a repository "
                "path containing Markdown, text, JSON, or YAML files."
            )

        dataset = build_dataset([str(path) for path in files], tags=tags or ["llama-models"])
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as fh:
            for row in dataset:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        return output_file
    finally:
        if cleanup_root and cleanup_root.exists():
            shutil.rmtree(cleanup_root, ignore_errors=True)
