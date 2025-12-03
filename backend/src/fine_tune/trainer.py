import openai
from typing import Any


class FineTuneEngine:
    def start_job(self, dataset_path: str) -> Any:
        res = openai.fine_tunes.create(training_file=dataset_path, model="gpt-4o-mini")
        return res["id"]
