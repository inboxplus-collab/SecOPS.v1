import json
import os
from typing import Dict, Iterable, List

from .dataset_builder import build_dataset
from .trainer import FineTuneEngine


class FineTuneJobScheduler:
    """Coordinates dataset creation and fine-tuning job submission."""

    def __init__(self, workspace: str = "data/fine_tune") -> None:
        self.workspace = workspace
        self.engine = FineTuneEngine()
        self.jobs: List[Dict[str, str]] = []
        os.makedirs(self.workspace, exist_ok=True)

    def schedule(self, name: str, files: Iterable[str]) -> str:
        dataset = build_dataset(files)
        dataset_path = os.path.join(self.workspace, f"{name}.jsonl")

        with open(dataset_path, "w", encoding="utf-8") as handle:
            for entry in dataset:
                handle.write(json.dumps(entry))
                handle.write("\n")

        job_id = self.engine.start_job(dataset_path)
        self.jobs.append({"name": name, "dataset": dataset_path, "job_id": job_id})
        return job_id

    def list_jobs(self) -> List[Dict[str, str]]:
        return list(self.jobs)
