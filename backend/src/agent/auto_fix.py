import os
import shutil
from typing import Optional


BACKUP_SUFFIX = ".secopsai.bak"


def auto_patch(file_path: Optional[str], patch_content: Optional[str]) -> None:
    """Apply a simple overwrite-based patch with a safety backup."""
    if not file_path or not patch_content:
        return

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target file not found: {file_path}")

    backup_path = f"{file_path}{BACKUP_SUFFIX}"
    shutil.copyfile(file_path, backup_path)

    with open(file_path, "w", encoding="utf-8") as target:
        target.write(patch_content)

    print(f"[SecOpsAI Agent] Applied patch to {file_path} (backup at {backup_path})")
