import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .dataset_builder import build_dataset
from .trainer import FineTuneEngine


class FineTuneJobScheduler:
    """Coordinates dataset creation and fine-tuning job submission."""

    def __init__(
        self,
        workspace: str = "data/fine_tune",
        seed_datasets: Sequence[str] | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

        if seed_datasets is None:
            builtin_seed = (
                Path(__file__).resolve().parents[2]
                / "data"
                / "fine_tune"
                / "security_hardening.jsonl"
            )
            self.seed_datasets = [builtin_seed]
        else:
            self.seed_datasets = [Path(p) for p in seed_datasets]

        self.engine = FineTuneEngine()
        self.jobs: List[Dict[str, str]] = []

    def _write_dataset(self, name: str, records: List[dict]) -> Path:
        dataset_path = self.workspace / f"{name}.jsonl"
        with dataset_path.open("w", encoding="utf-8") as handle:
            for entry in records:
                handle.write(json.dumps(entry))
                handle.write("\n")
        return dataset_path

    def schedule(self, name: str, files: Iterable[str]) -> str:
        user_dataset = build_dataset(files, tags=["user-provided"])

        seed_records: List[dict] = []
        for seed_path in self.seed_datasets:
            if seed_path.exists():
                if seed_path.suffix == ".jsonl":
                    with seed_path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            if line.strip():
                                record = json.loads(line)
                                record.setdefault("meta", {}).setdefault("tags", []).append("seed")
                                seed_records.append(record)
                else:
                    seed_records.extend(build_dataset([str(seed_path)], tags=["seed"]))

        dataset = seed_records + user_dataset
        dataset_path = self._write_dataset(name, dataset)

        job_id = self.engine.start_job(str(dataset_path))
        self.jobs.append(
            {
                "name": name,
                "dataset": str(dataset_path),
                "job_id": job_id,
            }
        )
        return job_id

    def list_jobs(self) -> List[Dict[str, str]]:
        return list(self.jobs)
