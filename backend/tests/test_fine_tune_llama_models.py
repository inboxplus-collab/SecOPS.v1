import importlib.util
import json
import sys
import types
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "fine_tune" / "llama_models_ingestor.py"
package = types.ModuleType("fine_tune")
package.__path__ = [str(MODULE_PATH.parent)]
sys.modules.setdefault("fine_tune", package)

spec = importlib.util.spec_from_file_location("fine_tune.llama_models_ingestor", MODULE_PATH)
assert spec and spec.loader
ingestor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingestor)

build_llama_models_dataset = ingestor.build_llama_models_dataset


def test_build_llama_models_dataset(tmp_path):
    repo_root = tmp_path / "llama-models"
    repo_root.mkdir()

    (repo_root / "README.md").write_text("Example vulnerability walkthrough", encoding="utf-8")
    (repo_root / "notes.txt").write_text("Extra training signal", encoding="utf-8")
    (repo_root / ".hidden.md").write_text("Should be ignored", encoding="utf-8")

    output_path = tmp_path / "dataset" / "llama_models.jsonl"
    dataset_file = build_llama_models_dataset(
        str(output_path),
        source_path=str(repo_root),
        tags=["llama-models", "test"],
    )

    assert dataset_file == output_path
    assert output_path.exists()

    rows = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 2

    entries = [json.loads(row) for row in rows]
    sources = {entry["meta"]["source"] for entry in entries}
    assert sources == {"README.md", "notes.txt"}
    assert all("SecOps AI" in entry["prompt"] for entry in entries)
    assert all(entry["meta"]["tags"] == ["llama-models", "test"] for entry in entries)
