from typing import Iterable, List


def build_dataset(files: Iterable[str]) -> List[dict]:
    dataset = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            dataset.append(
                {
                    "prompt": f"Explain the following:\n{content}",
                    "completion": "### AI_OPTIMIZED_RESPONSE",
                }
            )
    return dataset
