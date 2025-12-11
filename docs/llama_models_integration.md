# Using inboxplus-collab/llama-models for fine-tuning

The `fine_tune` package now includes helpers to convert the upstream
[`inboxplus-collab/llama-models`](https://github.com/inboxplus-collab/llama-models)
repository into a JSONL dataset that matches the SecOpsAI instruction format.

## Preparing the repository
- **Offline environments:** manually clone the repository and point the ingestor
  to the checkout via `source_path`.
- **Online environments:** the ingestor will automatically clone the repository
  using the default URL and branch.

```bash
# Manual clone when internet access is available
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/inboxplus-collab/llama-models.git ~/llama-models
```

## Building a dataset
```python
from fine_tune import build_llama_models_dataset

dataset_path = build_llama_models_dataset(
    "data/llama_models_dataset.jsonl",
    source_path="~/llama-models",  # omit to auto-clone when network is available
    tags=["llama-models", "external"],
)
print(f"Dataset written to {dataset_path}")
```

The ingestor walks the repository for Markdown, text, JSON, or YAML files and
wraps them with the SecOpsAI instruction prompt defined in `dataset_builder`.
Each entry includes metadata tags so the trainer can filter or bucket examples
as needed.

## Training
After the dataset is produced, pass it directly to `FineTuneEngine.start_job`:

```python
from fine_tune import FineTuneEngine

engine = FineTuneEngine(base_model="distilgpt2")
run_id = engine.start_job(dataset_path="data/llama_models_dataset.jsonl")
print(f"Fine-tune job completed: {run_id}")
```

The trainer saves the fine-tuned model and tokenizer under `models/fine_tuned/<run_id>`.
